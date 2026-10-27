"""The one clock every timestamp this app writes is read from: India time.

The people using the app are in India and read these times on the History
page, in Drive folder names and in the database. The server is not - EC2
runs on UTC - so a form scanned at 10:00 in a school office was recorded as
04:30, and one scanned just after midnight was filed under the previous
day. Every write now asks this module instead of the machine's own clock,
so it no longer matters where the code runs.

A fixed +05:30 offset rather than zoneinfo("Asia/Kolkata"): India has no
daylight saving, so the offset never changes, and zoneinfo needs the tzdata
package on Windows, which a laptop running the app may well not have.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "IST")


def now_ist() -> datetime:
    """The current time in India, as a NAIVE datetime.

    Naive on purpose: the columns are MySQL DATETIME, which stores no zone,
    and a naive value goes in exactly as written. It also compares cleanly
    with the naive values PyMySQL hands back when those columns are read.
    """
    return datetime.now(IST).replace(tzinfo=None)
