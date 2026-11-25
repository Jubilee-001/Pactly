/**
 * Closing a whole date (this issue's own slice of availability): a shop
 * that will not work on a given day must not leave a client's unpaid hold
 * sitting on that day's slot until its ten-minute clock runs out.
 *
 * A single day off must not mean deleting that day's slots and rebuilding
 * them next week, so a closure hides a *calendar date*; the slot rows stay
 * exactly where they are and reopening the date reveals them again. Nothing
 * here touches bookings: a date holding a locked booking is refused outright
 * (with the booking's id, so the panel can name it) rather than silently
 * cancelling anyone's appointment.
 *
 * No SQL lives here -- the two writes are `db/bookings.ts`'s
 * `releaseUnpaidHoldsForProviderDate` and `db/availabilitySlots.ts`'s
 * `closeSlotsInRange`.
 */
import { and, eq, gte, isNotNull, ne } from "drizzle-orm";

import {
  closeSlotsInRange,
  getSlotByProviderAndStart,
  listEarliestFutureSlotsByProvider,
  listOpenFutureSlots,
  listOpenSlotsInRange,
  type AvailabilitySlotRow,
} from "../db/availabilitySlots.js";
import { availabilitySlots, bookings } from "../db/schema.js";
import {
  releaseUnpaidHoldsForProviderDate,
  rescheduleBookingSlot,
} from "../db/bookings.js";
import type { Db } from "../db/client.js";
import { getProviderProfileById, getProviderProfileByWallet } from "../db/providerProfiles.js";
import {
  BookingEscrowStateError,
  SlotUnavailableError,
  getBookingForClient,
  getBookingView,
  type BookingView,
} from "./booking.js";
import { NotAProviderError } from "./profile.js";

const DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/;
const DAY_SECONDS = 24 * 60 * 60;
/** `Date#getTimezoneOffset`'s own documented range, in minutes. */
const MIN_TZ_OFFSET_MINUTES = -14 * 60;
const MAX_TZ_OFFSET_MINUTES = 14 * 60;

/** A locked booking can only move while its own free-cancellation window
 * (the existing `cancel_deadline`) is still open. */
export class BookingOutsideCancellationWindowError extends Error {
  constructor(message = "This booking is past its free-cancellation window and can no longer move.") {
    super(message);
    this.name = "BookingOutsideCancellationWindowError";
  }
}

export class InvalidCloseDateError extends Error {
  readonly details: Record<string, string>;
  constructor(details: Record<string, string>) {
    super("Pick a valid date to close.");
    this.name = "InvalidCloseDateError";
    this.details = details;
  }
}

export class InvalidBreaksError extends Error {
  readonly details: Record<string, string>;
  constructor(details: Record<string, string>) {
    super("One or more break values are invalid.");
    this.name = "InvalidBreaksError";
    this.details = details;
  }
}

export interface CloseProviderDateInput {
  /** Local calendar date, `YYYY-MM-DD`. */
  date: string;
  /** `new Date().getTimezoneOffset()` from the caller's browser -- the same
   * signed-minutes convention `POST /bookings/hold` already uses. Omitted
   * means UTC. */
  tzOffsetMinutes?: number;
}

export interface CloseProviderDateView {
  date: string;
  /** UTC epoch seconds of the local day's start/end, echoed so a caller can
   * check exactly which window was closed. */
  rangeStart: number;
  rangeEnd: number;
  releasedHoldCount: number;
  releasedBookingIds: string[];
  /** Future open slots removed from this date's availability. */
  closedSlotCount: number;
  /** Funded bookings left alone -- the shop resolves these from Bookings. */
  needsResolution: Array<{ bookingId: string; slotStartsAt: number }>;
}

/**
 * Turns a local `YYYY-MM-DD` plus the browser's offset into the UTC epoch
 * window`[start, end)`. Offset sign matches `getTimezoneOffset`: it is UTC
 * minus local, so local midnight is `Date.UTC(...) + offset * 60_000`.
 * Throws {@link InvalidCloseDateError} for a malformed or impossible date,
 * or a non-integer/out-of-range offset -- never a raw `Date` NaN.
 */
export function localDateRange(
  date: string,
  tzOffsetMinutes = 0,
): { rangeStart: number; rangeEnd: number } {
  const match = DATE_PATTERN.exec(date);
  if (!match) {
    throw new InvalidCloseDateError({ date: "Use a calendar date in YYYY-MM-DD form." });
  }
  if (
    !Number.isInteger(tzOffsetMinutes) ||
    tzOffsetMinutes < MIN_TZ_OFFSET_MINUTES ||
    tzOffsetMinutes > MAX_TZ_OFFSET_MINUTES
  ) {
    throw new InvalidCloseDateError({ tzOffsetMinutes: "Timezone offset must be whole minutes between -840 and 840." });
  }
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const utcMidnight = Date.UTC(year, month - 1, day);
  const check = new Date(utcMidnight);
  if (check.getUTCFullYear() !== year || check.getUTCMonth() !== month - 1 || check.getUTCDate() !== day) {
    throw new InvalidCloseDateError({ date: "That date does not exist." });
  }
  const rangeStart = Math.floor(utcMidnight / 1000) + tzOffsetMinutes * 60;
  return { rangeStart, rangeEnd: rangeStart + DAY_SECONDS };
}

/**
 * `POST /bookings/:id/reschedule`: moves the caller's own locked booking to
 * another open slot of the same shop, inside the booking's free-cancellation
 * window. Only `slot_id` and `cancel_deadline` are written -- the escrow
 * contract id, deposit amounts, client and provider are untouched, so the
 * deposit keeps sitting in the same contract and no second escrow is
 * created. The old slot simply stops being booked (it becomes available);
 * the new slot becomes this booking's slot.
 *
 * The target is looked up by this booking's own `provider_profile_id`, so a
 * different shop's slot can never match, and both slots share the one
 * `sessionLengthMinutes` that belongs to that shop -- the duration is
 * therefore the same by construction. A slot taken between the caller's
 * click and this write makes the atomic DB move return `false`, which is
 * surfaced as {@link SlotUnavailableError}; nothing is overwritten.
 */
export async function rescheduleLockedBooking(
  db: Db,
  bookingId: string,
  walletAddress: string,
  newSlotStartsAt: number,
  now: number = Math.floor(Date.now() / 1000),
): Promise<BookingView> {
  const booking = await getBookingForClient(db, bookingId, walletAddress);
  if (booking.escrowState !== "locked") {
    throw new BookingEscrowStateError("Only a booking with a locked deposit can move to another time.");
  }
  if (now >= booking.cancelDeadline) {
    throw new BookingOutsideCancellationWindowError();
  }
  if (!booking.slotId) {
    throw new SlotUnavailableError("This booking has no slot to move from.");
  }
  if (!Number.isInteger(newSlotStartsAt)) {
    throw new SlotUnavailableError("Pick a valid slot start time.");
  }

  const newSlot = await getSlotByProviderAndStart(db, booking.providerProfileId, newSlotStartsAt);
  if (!newSlot || newSlot.withdrawnAt !== null || newSlot.startsAt <= now) {
    throw new SlotUnavailableError("That time is no longer open.");
  }
  if (newSlot.id === booking.slotId) {
    return getBookingView(db, bookingId, walletAddress);
  }

  const profile = await getProviderProfileById(db, booking.providerProfileId);
  if (!profile) {
    // Unreachable: `provider_profile_id` is a foreign key.
    throw new SlotUnavailableError("That shop no longer has a profile.");
  }
  const newCancelDeadline = newSlot.startsAt - profile.cancellationWindowHours * 3600;

  const moved = await rescheduleBookingSlot(db, bookingId, booking.slotId, newSlot.id, newCancelDeadline, now);
  if (!moved) {
    throw new SlotUnavailableError("That time was taken just now. Pick another open time.");
  }
  return getBookingView(db, bookingId, walletAddress);
}

/**
 * `POST /me/provider/availability/close-date`: resolves the caller's own
 * provider profile, releases the date's active unpaid holds (recording
 * `shop-closed-date` on them so their clients see why), then removes that
 * date's future open slots. Other dates are untouched. Idempotent: closing
 * an already-closed or hold-free date succeeds and reports zero releases.
 */
export async function closeProviderDate(
  db: Db,
  walletAddress: string,
  input: CloseProviderDateInput,
  now: number = Math.floor(Date.now() / 1000),
): Promise<CloseProviderDateView> {
  const profile = await getProviderProfileByWallet(db, walletAddress);
  if (!profile) {
    throw new NotAProviderError();
  }
  const { rangeStart, rangeEnd } = localDateRange(input.date, input.tzOffsetMinutes ?? 0);

  const release = await releaseUnpaidHoldsForProviderDate(db, profile.id, rangeStart, rangeEnd, now);
  const closedSlotCount = await closeSlotsInRange(db, profile.id, rangeStart, rangeEnd, now);

  return {
    date: input.date,
    rangeStart,
    rangeEnd,
    releasedHoldCount: release.released.length,
    releasedBookingIds: release.released.map((booking) => booking.bookingId),
    closedSlotCount,
    needsResolution: release.needsResolution.map((booking) => ({
      bookingId: booking.bookingId,
      slotStartsAt: booking.slotStartsAt,
    })),
  };
}

/** Options for {@link listOfferedSlots}. */
export interface ListOfferedSlotsOptions {
  /** Injectable clock (UTC epoch seconds). */
  now?: number;
  /** Caps the window to `now + withinSeconds`. */
  withinSeconds?: number;
  /** Issue #36: minimum notice in minutes -- slots sooner than this are excluded. */
  minimumNoticeMinutes?: number;
  /** The shop's own session length, used for buffer math. */
  sessionLengthMinutes: number;
  /** Minutes kept free after a locked booking. Zero means no buffer filtering. */
  bufferMinutes: number;
}

/** `listOpenFutureSlots` plus post-lock buffer filtering -- returns the
 * `startsAt` values of slots a client can actually hold. Slots that start
 * inside the buffer period after any active booking are excluded. */
export async function listOfferedSlots(
  db: Db,
  providerProfileId: string,
  options: ListOfferedSlotsOptions,
): Promise<number[]> {
  const now = options.now ?? Math.floor(Date.now() / 1000);
  const rows = await listOpenFutureSlots(db, providerProfileId, {
    now,
    withinSeconds: options.withinSeconds,
    minimumNoticeMinutes: options.minimumNoticeMinutes,
  });
  if (options.bufferMinutes <= 0) {
    return rows.map((row) => row.startsAt);
  }
  const lockedStarts = await listLockedBookingStarts(db, providerProfileId);
  const sessionLengthMs = options.sessionLengthMinutes * 60;
  const bufferMs = options.bufferMinutes * 60;
  return rows
    .filter((row) => {
      return !lockedStarts.some((lockedStart) => {
        const sessionEnd = lockedStart + sessionLengthMs;
        const bufferEnd = sessionEnd + bufferMs;
        return row.startsAt > lockedStart && row.startsAt < bufferEnd;
      });
    })
    .map((row) => row.startsAt);
}

/** Earliest open slots per provider, with per-provider buffer filtering. */
export async function listEarliestOfferedSlotsByProvider(
  db: Db,
  providers: Array<{ id: string; sessionLengthMinutes: number; bufferMinutes: number }>,
  now: number = Math.floor(Date.now() / 1000),
  limit: number = 3,
): Promise<Map<string, number[]>> {
  const result = new Map<string, number[]>();
  if (providers.length === 0) return result;
  const providerIds = providers.map((p) => p.id);
  const byProvider = await listEarliestFutureSlotsByProvider(db, providerIds, { now, limit });
  for (const provider of providers) {
    let slots = byProvider.get(provider.id) ?? [];
    if (provider.bufferMinutes > 0) {
      const lockedStarts = await listLockedBookingStarts(db, provider.id);
      const sessionLengthMs = provider.sessionLengthMinutes * 60;
      const bufferMs = provider.bufferMinutes * 60;
      slots = slots.filter((slotStart) => {
        return !lockedStarts.some((lockedStart) => {
          const sessionEnd = lockedStart + sessionLengthMs;
          const bufferEnd = sessionEnd + bufferMs;
          return slotStart > lockedStart && slotStart < bufferEnd;
        });
      });
    }
    result.set(provider.id, slots);
  }
  return result;
}

/** Open slots within `[rangeStart, rangeEnd)` (half-open, UTC epoch seconds),
 * excluding slots taken by active bookings. Returns `startsAt` values. */
export async function listOfferedSlotsInRange(
  db: Db,
  providerProfileId: string,
  rangeStart: number,
  rangeEnd: number,
  now: number = Math.floor(Date.now() / 1000),
): Promise<number[]> {
  return listOpenSlotsInRange(db, providerProfileId, rangeStart, rangeEnd, now);
}

/** Start times of every slot that currently carries an active booking
 * (a non-expired hold, or a locked/released booking) for this provider. */
export async function listLockedBookingStarts(db: Db, providerProfileId: string): Promise<number[]> {
  const now = Math.floor(Date.now() / 1000);
  const rows = await db
    .select({ startsAt: availabilitySlots.startsAt })
    .from(bookings)
    .innerJoin(availabilitySlots, eq(bookings.slotId, availabilitySlots.id))
    .where(
      and(
        eq(bookings.providerProfileId, providerProfileId),
        isNotNull(bookings.slotId),
        or(
          and(isNotNull(bookings.escrowState), ne(bookings.escrowState, "refunded")),
          and(isNull(bookings.escrowState), gte(bookings.holdExpiresAt, now)),
        ),
      ),
    );
  return rows.map((row) => row.startsAt);
}

/** True when `slotStartsAt` starts inside the post-lock buffer period after
 * a locked booking that begins at `lockedStartsAt`. Zero buffer means never
 * inside the buffer. */
import { zonedClock } from "./timezone.js";

/** A break cut out of one weekday's working hours. `weekday` is 0 (Sunday)
 * through 6 (Saturday) in the shop's own timezone; the window is
 * `[startMinute, endMinute)`. */
export interface ProviderBreak {
  weekday: number;
  startMinute: number;
  endMinute: number;
}

/** Validates raw break input and returns a normalized array of breaks.
 * Throws {@link InvalidBreaksError} for invalid input. */
export function validateBreaks(rawBreaks: unknown): ProviderBreak[] {
  if (!Array.isArray(rawBreaks)) {
    throw new InvalidBreaksError({ breaks: "breaks must be an array" });
  }
  const breaks: ProviderBreak[] = [];
  const seen = new Set<string>();
  for (const raw of rawBreaks) {
    if (
      !raw ||
      typeof raw !== "object" ||
      typeof (raw as { weekday?: unknown }).weekday !== "number" ||
      typeof (raw as { startMinute?: unknown }).startMinute !== "number" ||
      typeof (raw as { endMinute?: unknown }).endMinute !== "number"
    ) {
      throw new InvalidBreaksError({ breaks: "each break must have weekday, startMinute and endMinute" });
    }
    const { weekday, startMinute, endMinute } = raw as { weekday: number; startMinute: number; endMinute: number };
    if (!Number.isInteger(weekday) || weekday < 0 || weekday > 6) {
      throw new InvalidBreaksError({ weekday: "weekday must be 0 (Sunday) through 6 (Saturday)" });
    }
    if (!Number.isInteger(startMinute) || startMinute < 0 || startMinute >= 1440) {
      throw new InvalidBreaksError({ startMinute: "startMinute must be a whole number between 0 and 1439" });
    }
    if (!Number.isInteger(endMinute) || endMinute <= 0 || endMinute > 1440) {
      throw new InvalidBreaksError({ endMinute: "endMinute must be a whole number between 1 and 1440" });
    }
    if (startMinute >= endMinute) {
      throw new InvalidBreaksError({ startMinute: "startMinute must be before endMinute" });
    }
    const key = `${weekday}-${startMinute}-${endMinute}`;
    if (seen.has(key)) {
      throw new InvalidBreaksError({ breaks: "duplicate break" });
    }
    seen.add(key);
    breaks.push({ weekday, startMinute, endMinute });
  }
  return breaks;
}

/** Computes the working range for each weekday from a set of slot start times.
 * For each weekday, the range is [earliest slot start, latest slot start + sessionLength). */
export function workingRangesForSlots(
  slots: number[],
  sessionLengthMinutes: number,
  timeZone: string,
): Array<{ weekday: number; startMinute: number; endMinute: number }> {
  const ranges = new Map<number, { start: number; end: number }>();
  for (const startsAt of slots) {
    const { weekday, minuteOfDay } = zonedClock(startsAt, timeZone);
    const end = minuteOfDay + sessionLengthMinutes;
    const existing = ranges.get(weekday);
    if (!existing) {
      ranges.set(weekday, { start: minuteOfDay, end });
    } else {
      existing.start = Math.min(existing.start, minuteOfDay);
      existing.end = Math.max(existing.end, end);
    }
  }
  return Array.from(ranges.entries()).map(([weekday, range]) => ({
    weekday,
    startMinute: range.start,
    endMinute: range.end,
  }));
}

/** Asserts that every break lies inside its weekday's working range and that
 * breaks on the same weekday do not overlap. Throws {@link InvalidBreaksError}. */
export function assertBreaksWithinRanges(
  breaks: ProviderBreak[],
  ranges: Array<{ weekday: number; startMinute: number; endMinute: number }>,
): void {
  const rangeByWeekday = new Map<number, { startMinute: number; endMinute: number }>();
  for (const range of ranges) {
    rangeByWeekday.set(range.weekday, range);
  }
  for (const brk of breaks) {
    const range = rangeByWeekday.get(brk.weekday);
    if (!range) {
      throw new InvalidBreaksError({ weekday: `no working hours on weekday ${brk.weekday}` });
    }
    if (brk.startMinute < range.startMinute || brk.endMinute > range.endMinute) {
      throw new InvalidBreaksError({ weekday: `break on weekday ${brk.weekday} is outside working hours` });
    }
  }
  const byWeekday = new Map<number, ProviderBreak[]>();
  for (const brk of breaks) {
    const list = byWeekday.get(brk.weekday) ?? [];
    list.push(brk);
    byWeekday.set(brk.weekday, list);
  }
  for (const [, list] of byWeekday) {
    list.sort((a, b) => a.startMinute - b.startMinute);
    for (let i = 1; i < list.length; i++) {
      if (list[i].startMinute < list[i - 1].endMinute) {
        throw new InvalidBreaksError({ weekday: `breaks on weekday ${list[0].weekday} overlap` });
      }
    }
  }
}

/** Removes slots that start inside any break period. Breaks are named in the
 * shop's own wall clock; slots are UTC epoch seconds. */
export function filterSlotsOutsideBreaks(
  slots: number[],
  timeZone: string,
  breaks: ProviderBreak[],
): number[] {
  if (breaks.length === 0) return slots;
  return slots.filter((startsAt) => {
    const { weekday, minuteOfDay } = zonedClock(startsAt, timeZone);
    return !breaks.some(
      (brk) => brk.weekday === weekday && brk.startMinute <= minuteOfDay && minuteOfDay < brk.endMinute,
    );
  });
}

/** True when `slotStartsAt` starts inside the post-lock buffer period after
 * a locked booking that begins at `lockedStartsAt`. Zero buffer means never
 * inside the buffer. */
export function isSlotInsideBufferAfterLocked(
  slotStartsAt: number,
  lockedStartsAt: number,
  sessionLengthMinutes: number,
  bufferMinutes: number,
): boolean {
  if (bufferMinutes <= 0) return false;
  const sessionEnd = lockedStartsAt + sessionLengthMinutes * 60;
  const bufferEnd = sessionEnd + bufferMinutes * 60;
  return slotStartsAt >= sessionEnd && slotStartsAt < bufferEnd;
}
