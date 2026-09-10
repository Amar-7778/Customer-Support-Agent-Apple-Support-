"""
Heuristics module for evaluating conversation thread resolution status.

This module provides configurable rules to tag conversation threads as:
- resolved: "true" | "false" | "unknown"
along with an explicit resolution_reason explaining why the heuristic made this determination.

NOTE: This is documented as a heuristic approximation, not ground truth.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

DEFAULT_GRATITUDE_KEYWORDS: Tuple[str, ...] = (
    "thank",
    "thanks",
    "thx",
    "thankyou",
    "thank you",
    "appreciate",
    "appreciated",
    "resolved",
    "fixed",
    "working now",
    "works now",
    "sorted",
    "all good",
    "all set",
    "great help",
    "helpful",
    "much obliged",
    "kudos",
    "cheers",
)


@dataclass(frozen=True)
class ResolutionConfig:
    """Configuration for thread resolution heuristic."""

    inactivity_hours: float = 24.0
    gratitude_keywords: Tuple[str, ...] = DEFAULT_GRATITUDE_KEYWORDS
    case_sensitive: bool = False

    def compile_regex(self) -> re.Pattern:
        """Compile gratitude keyword regex with word boundaries."""
        escaped = [re.escape(k.strip()) for k in self.gratitude_keywords if k.strip()]
        # Pattern matches word boundaries or start/end of string for phrases
        pattern = r"(?:\b(?:{}))\b".format("|".join(escaped))
        flags = 0 if self.case_sensitive else re.IGNORECASE
        return re.compile(pattern, flags)


def has_closure_language(text: Optional[str], config: ResolutionConfig) -> bool:
    """Check if text contains closure or gratitude keywords."""
    if not text or not isinstance(text, str):
        return False
    regex = config.compile_regex()
    return bool(regex.search(text))


def evaluate_thread_resolution(
    tweets: List[Dict[str, Any]],
    brand_handle: Optional[str] = None,
    dataset_max_time: Optional[datetime] = None,
    config: Optional[ResolutionConfig] = None,
) -> Tuple[str, str]:
    """
    Evaluate resolution status of an ordered conversation thread.

    Parameters
    ----------
    tweets : List[Dict[str, Any]]
        List of tweet dicts in chronological order.
        Each dict must contain:
        - "inbound": bool (True = customer, False = brand/agent)
        - "author_id": str
        - "created_at": datetime or ISO string
        - "text": str
    brand_handle : Optional[str]
        Specific brand handle being evaluated (e.g. "AppleSupport").
        If provided, outbound messages matching this handle are prioritized.
    dataset_max_time : Optional[datetime]
        Max timestamp present in the full dataset to evaluate dormancy.
    config : Optional[ResolutionConfig]
        Configuration instance for inactivity threshold and gratitude keywords.

    Returns
    -------
    Tuple[str, str]
        (resolved_tag, resolution_reason)
        resolved_tag is "true", "false", or "unknown"
    """
    if config is None:
        config = ResolutionConfig()

    if not tweets:
        return "unknown", "empty_thread"

    # Fast date parsing
    def parse_dt(val: Any) -> Optional[datetime]:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
        if isinstance(val, str):
            try:
                return datetime.strptime(val, "%a %b %d %H:%M:%S +0000 %Y").replace(tzinfo=timezone.utc)
            except Exception:
                pass
        try:
            dt = pd.to_datetime(val)
            if hasattr(dt, "to_pydatetime"):
                dt = dt.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    parsed_tweets = []
    for tw in tweets:
        parsed_tweets.append({
            "inbound": bool(tw.get("inbound", False)),
            "author_id": str(tw.get("author_id", "")),
            "text": str(tw.get("text", "") or ""),
            "created_at": tw.get("created_at"),
        })

    # Separate inbound (customer) and outbound (brand) tweets
    customer_tweets = [t for t in parsed_tweets if t["inbound"]]
    brand_tweets = [
        t for t in parsed_tweets
        if (not t["inbound"]) or (brand_handle and t["author_id"].lower() == brand_handle.lower())
    ]

    # Edge Case 1: Customer tweeted, brand never replied
    if customer_tweets and not brand_tweets:
        return "false", "unanswered_by_brand"

    # Edge Case 2: Only brand tweets present (outbound announcement or unlinked message)
    if brand_tweets and not customer_tweets:
        return "unknown", "single_sided_outbound_announcement"

    # Both customer and brand tweets are present. Inspect the final tweet in the thread.
    last_tweet = parsed_tweets[-1]
    last_customer_tweet = customer_tweets[-1]

    # Check if the customer expressed gratitude / closure in their last message
    customer_gratitude = has_closure_language(last_customer_tweet["text"], config)

    if last_tweet["inbound"]:
        # The customer posted the final message in the thread
        if customer_gratitude:
            return "true", "customer_gratitude_closure"
        else:
            # Customer asked a question or responded and is waiting for brand reply
            return "false", "pending_brand_reply"
    else:
        # The brand posted the final message in the thread
        if customer_gratitude:
            # Customer thanked the brand, and brand gave a final signoff
            return "true", "customer_gratitude_confirmed_by_brand"

        # Check dormancy / inactivity
        last_tweet_dt = parse_dt(last_tweet["created_at"])
        if last_tweet_dt and dataset_max_time:
            ds_max = parse_dt(dataset_max_time)
            if ds_max:
                hours_since_brand_reply = (ds_max - last_tweet_dt).total_seconds() / 3600.0
                if hours_since_brand_reply >= config.inactivity_hours:
                    return "true", f"brand_final_reply_dormant_{config.inactivity_hours:.0f}h"
                else:
                    return "unknown", "brand_final_reply_recent_insufficient_observation_window"

        # If no dataset max time provided, brand replied last with no customer rebuttal
        return "true", f"brand_final_reply_dormant_{config.inactivity_hours:.0f}h"
