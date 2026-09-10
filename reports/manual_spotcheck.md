# Manual Thread Spot-Check Report (15 Sampled Threads)

**Date**: 2026-09-10  
**Dataset**: `data/processed/AppleSupport_threads.parquet`  
**Sampling Seed**: 42 (Stratified: 5 `brand_final_reply_dormant_24h`, 5 gratitude-tagged, 5 unresolved/unknown)

---

## Executive Summary & Flag

> [!WARNING]
> **FLAGGED: HEURISTIC LIMITATION DETECTED (4/15 Tagging Discrepancies)**  
> 4 out of the 15 sampled threads exhibited clear resolution-tagging errors or nuances under human review:
> 1. **Thread 7 (T_543708)**: False positive gratitude tag caused by keyword `"fixed"` inside a conditional complaint (*"if this isn't fixed soon..."*).
> 2. **Thread 8 (T_663919)**: False positive gratitude tag caused by keyword `"fixed"` inside a past-tense complaint (*"I fixed the I thing once before, I'm not doing it again"*).
> 3. **Thread 9 (T_1862049)**: False positive gratitude tag caused by sarcastic keyword `"Cheers"` (*"Cheers @AppleSupport for deleting all my notes! Not the only thing that's happened either 😒"*).
> 4. **Thread 13 (T_360269)**: False negative unresolved tag where brand resolved the question and user thanked Apple (*"Thank you Apple!!"*), but subsequent customer-to-customer banter caused the thread to terminate on a customer tweet, classifying it as `pending_brand_reply`.
>
> **Reconstruction Order Coherence**: **15/15 (100%) Coherent**. All reply graphs preserved chronological ordering and valid conversation chains without mismatched branches or broken trees.

---

## Detailed Thread Audits

### Stratum A: Tagged `brand_final_reply_dormant_24h` (`resolved: true`)

#### 1. Thread `T_1895458` (Length: 2, Branching: False)
- **Turn 1 (Customer 564969)** | `Tue Oct 24 04:48:07 +0000 2017`:  
  > *"@AppleSupport Since the 11.0.3 update, my ipad screen is so dark! I can barely read it during the day even on the brightest setting."*
- **Turn 2 (Brand AppleSupport)** | `Tue Oct 24 12:54:32 +0000 2017`:  
  > *"@564969 We've got your back We’ll be glad to look into this for you. Can you start by confirming you iOS version via DM https://t.co/GDrqU22YpT"*
- **Order Coherence**: **Yes**. Chronological customer complaint followed by brand response.
- **Resolution Tag Judgment**: **Acceptable under Heuristic / Unclear in Reality**. Apple invited the user to DM. The public thread went dormant (>24 hours). We cannot confirm from public data whether the issue was resolved in DM, but this aligns with the documented dormancy heuristic.

---

#### 2. Thread `T_1895491` (Length: 2, Branching: False)
- **Turn 1 (Customer 564980)** | `Tue Oct 24 05:27:49 +0000 2017`:  
  > *"@AppleSupport I think y'all sent me a fake pair of headphones."*
- **Turn 2 (Brand AppleSupport)** | `Tue Oct 24 12:55:33 +0000 2017`:  
  > *"@564980 We can definitely look into this with you. Send us a DM and let us know what type of headphones you received. https://t.co/GDrqU22YpT"*
- **Order Coherence**: **Yes**. Perfect chronological turn.
- **Resolution Tag Judgment**: **Acceptable under Heuristic / Unclear in Reality**. Standard brand redirection to DM followed by public silence.

---

#### 3. Thread `T_199307` (Length: 2, Branching: False)
- **Turn 1 (Customer 162972)** | `Wed Oct 04 12:17:53 +0000 2017`:  
  > *"iPhone 6s is officially one year old today... and my Touch ID just randomly stopped working @115858 #helpme #forgotallmypasswords"*
- **Turn 2 (Brand AppleSupport)** | `Wed Oct 04 14:27:34 +0000 2017`:  
  > *"@162972 We want to help. Do you know the passcode for the iPhone? We can use that to unlock for right now and troubleshoot."*
- **Order Coherence**: **Yes**. Direct prompt and troubleshooting reply.
- **Resolution Tag Judgment**: **Yes (Dormant)**. Apple asked an active diagnostic question; customer did not return to Twitter for >24h.

---

#### 4. Thread `T_236840` (Length: 2, Branching: False)
- **Turn 1 (Customer 172404)** | `Thu Oct 05 09:47:37 +0000 2017`:  
  > *"@AppleSupport @118936 iPhone Won’t Turn-on After Upgrade to iOS 11 \npoor quality OS provided by apple.Don't purchase and upgrade iphone"*
- **Turn 2 (Brand AppleSupport)** | `Thu Oct 05 14:16:00 +0000 2017`:  
  > *"@172404 Hi. We're here for you. Have you tried the steps here: https://t.co/8MvHMFZ83h Let us know."*
- **Order Coherence**: **Yes**. Immediate topical response with relevant knowledge base link.
- **Resolution Tag Judgment**: **Yes (Likely Resolved)**. Direct official troubleshooting link provided for black screen/power failure; customer did not follow up.

---

#### 5. Thread `T_2111447` (Length: 2, Branching: False)
- **Turn 1 (Customer 622792)** | `Wed Nov 08 12:27:24 +0000 2017`:  
  > *"Come on @AppleSupport sort out the numerous bugs affecting iPhone 6 users with ios11, it’s getting ridiculous 😡"*
- **Turn 2 (Brand AppleSupport)** | `Wed Nov 08 16:26:00 +0000 2017`:  
  > *"@622792 We're glad you've reached out. Please send us a DM with some details on what you're running into. We're eager to help. https://t.co/GDrqU22YpT"*
- **Order Coherence**: **Yes**. Coherent exchange.
- **Resolution Tag Judgment**: **Acceptable under Heuristic / Unclear in Reality**. Customer vented; Apple requested DM details. No public follow-up.

---

### Stratum B: Tagged via Gratitude (`customer_gratitude_confirmed_by_brand` / `customer_gratitude_closure`)

#### 6. Thread `T_2207555` (Length: 4, Branching: False)
- **Turn 1 (Customer 645410)** | `Thu Nov 09 22:34:52 +0000 2017`:  
  > *"KAY @115858 CAN WE GET THIS I️ SHIT FIXED ORRRRRRRRR"*
- **Turn 2 (Brand AppleSupport)** | `Fri Nov 10 00:07:00 +0000 2017`:  
  > *"@645410 We absolutely can. We released iOS 11.1.1 to fix the auto-correct issue today. If you haven't already, make sure you have a current backup and update. This article explains: https://t.co/qODbOsp4wz"*
- **Turn 3 (Customer 645410)** | `Fri Nov 10 01:28:17 +0000 2017`:  
  > *"@AppleSupport Wow, thanks. Now I feel bad for swearing."*
- **Turn 4 (Brand AppleSupport)** | `Fri Nov 10 02:29:00 +0000 2017`:  
  > *"@645410 You're welcome. Feel free to reach out to us if you have any further questions. Have a wonderful day!"*
- **Order Coherence**: **Yes (Exemplary)**. Perfectly constructed 4-turn resolution arc.
- **Resolution Tag Judgment**: **Yes (100% Valid)**. Grounded resolution: software patch identified, customer expresses genuine gratitude, brand confirms closure.

---

#### 7. Thread `T_543708` (Length: 2, Branching: False)
- **Turn 1 (Customer 246296)** | `Sat Dec 02 08:22:26 +0000 2017`:  
  > *".@AppleSupport @115858 - how come there isn't already information here about the widely reported \"2nd December\" bug - my phone keeps crashing/re-starting. I'm due an upgrade and so if this isn't fixed soon and fully I'll be choosing @122986 #Pixel2 Poor........."*
- **Turn 2 (Brand AppleSupport)** | `Sat Dec 02 16:27:00 +0000 2017`:  
  > *"@246296 We want your iPhone to function properly for you. Check out this support article, which addresses the issue you're experiencing: https://t.co/Wc5MHFZb0p"*
- **Order Coherence**: **Yes**. Coherent complaint and brand response.
- **Resolution Tag Judgment**: **FLAGGED: FALSE POSITIVE GRATITUDE**.
  - Heuristic Tag: `customer_gratitude_confirmed_by_brand`
  - Human Review: The customer was complaining: *"if this isn't fixed soon..."*. The regex matched `"fixed"` from the gratitude dictionary. The customer never expressed gratitude. (It could still be considered resolved under dormancy since >24h elapsed, but the assigned reason tag is factually wrong).

---

#### 8. Thread `T_663919` (Length: 2, Branching: False)
- **Turn 1 (Customer 278224)** | `Thu Nov 23 02:02:54 +0000 2017`:  
  > *"I fixed the I thing once before, I’m not doing it again. Your move, @115858"*
- **Turn 2 (Brand AppleSupport)** | `Thu Nov 23 02:14:00 +0000 2017`:  
  > *"@278224 Please be sure your device is updated to iOS 11.1.2. If you need to update, be sure to back up first using these steps: https://t.co/4f8hwT5to6 If the issue is persisting, you can shoot us a DM. From there we can look into things more with you. https://t.co/GDrqU22YpT"*
- **Order Coherence**: **Yes**. Clear exchange.
- **Resolution Tag Judgment**: **FLAGGED: FALSE POSITIVE GRATITUDE**.
  - Heuristic Tag: `customer_gratitude_confirmed_by_brand`
  - Human Review: Customer statement *"I fixed the I thing once before, I'm not doing it again"* used `"fixed"` to describe prior manual workaround fatigue, not resolution gratitude.

---

#### 9. Thread `T_1862049` (Length: 2, Branching: False)
- **Turn 1 (Customer 556497)** | `Thu Oct 19 17:47:30 +0000 2017`:  
  > *"Cheers @AppleSupport for deleting all my notes! Not the only thing that's happened either 😒 Think it's time to switch to @125607 📱"*
- **Turn 2 (Brand AppleSupport)** | `Thu Oct 19 18:11:31 +0000 2017`:  
  > *"@556497 Having those notes ready when you need them is important, and we want to get this resolved for you. DM us and we'll get started. https://t.co/GDrqU22YpT"*
- **Order Coherence**: **Yes**. Coherent context.
- **Resolution Tag Judgment**: **FLAGGED: FALSE POSITIVE GRATITUDE (SARCASM)**.
  - Heuristic Tag: `customer_gratitude_confirmed_by_brand`
  - Human Review: Customer used *"Cheers"* sarcastically to complain about lost data. The regex matched `"cheers"` as positive gratitude.

---

#### 10. Thread `T_1149745` (Length: 13, Branching: True)
- **Turn 1 (Customer 390558)** | `Tue Oct 24 13:46:14`: *"@AppleSupport New apple id, cant review it for itunes store, cant post on communities to gelp help, Could you lend me a hand?"*
- **Turn 2 (Brand AppleSupport)** | `Tue Oct 24 14:48:30`: *"@390558 We'll be more than happy to help you today with your Apple ID. Let us know what's happening."*
- **Turn 3 (Customer 390558)** | `Tue Oct 24 14:51:36`: *"@AppleSupport When I launch appstore it asks me to review my acc, but when I press review I get a blank page. Tried on PC too."*
- **Turn 4 (Brand AppleSupport)** | `Tue Oct 24 15:28:29`: *"@390558 Let's try restarting the computer and see if you can review the account screen from there."*
- **Turn 5 (Customer 390558)** | `Tue Oct 24 15:29:16`: *"@AppleSupport Restarted the computer &amp; the phone - the issue persists."*
- **Turn 6 (Customer 390559 - Branch)** | `Tue Oct 24 15:37:58`: *"@390558 @AppleSupport Blank screen on both phone and iTunes on pc https://t.co/UD3Di3Lu92"*
- **Turn 7 (Brand AppleSupport)** | `Tue Oct 24 15:50:46`: *"@390558 Thank you. To confirm, you aren't able to access iTunes Store, or the Communities sites? Please DM, we'll meet you there to help. https://t.co/GDrqU22YpT"*
- **Turn 8 (Brand AppleSupport - to Branch)** | `Tue Oct 24 15:59:48`: *"@390559 We'd love to lend a hand! Which iPhone are you using, what iOS version is on it and when did this issue begin?"*
- **Turn 9 (Customer 390559)** | `Tue Oct 24 16:05:05`: *"@AppleSupport iPhone 7 iOS 11.0.3. brand new phone and account"*
- **Turn 10 (Customer 394437 - 2nd Branch)** | `Tue Oct 24 18:13:55`: *"@390558 @AppleSupport Same for me @AppleSupport , please send us an update!"*
- **Turn 11 (Brand AppleSupport)** | `Tue Oct 24 18:37:31`: *"@394437 We'd love to look into this. Please click here to join us in DM. We'll pick up there. https://t.co/GDrqU22YpT"*
- **Turn 12 (Customer 394437)** | `Tue Oct 24 18:46:45`: *"@AppleSupport At the time I published this tweet, the problem was solved. Thanks for your quick reply."*
- **Turn 13 (Brand AppleSupport)** | `Tue Oct 24 18:59:01`: *"@394437 You're very welcome."*
- **Order Coherence**: **Yes**. Highly complex 13-tweet branching graph reconstructed seamlessly with chronological order maintained.
- **Resolution Tag Judgment**: **Yes (Valid)**. Customer 394437 explicitly confirmed the issue was solved with thanks, and brand replied with acknowledgment.

---

### Stratum C: Tagged Unresolved / Unknown (`pending_brand_reply` / `brand_final_reply_recent_insufficient_observation_window`)

#### 11. Thread `T_1632363` (Length: 3, Branching: False)
- **Turn 1 (Customer 499503)** | `Sun Nov 05 18:38:09 +0000 2017`:  
  > *"@AppleSupport My Situation Has Been Going On For 3 DAYS, Thanks To U... #Let’sGetThisDONE!!!!!"*
- **Turn 2 (Brand AppleSupport)** | `Sun Nov 05 20:08:33 +0000 2017`:  
  > *"@499503 Hi there. We've received your DM and shared some tips to help there. Please watch for our response shortly."*
- **Turn 3 (Customer 499503)** | `Sun Nov 05 20:09:39 +0000 2017`:  
  > *"@AppleSupport I Repiled..."*
- **Order Coherence**: **Yes**. Clear chronological sequence.
- **Resolution Tag Judgment**: **Yes (Accurate)**. The conversation ends on a customer message waiting for brand action. Correctly tagged `pending_brand_reply` (`resolved: false`).

---

#### 12. Thread `T_1443310` (Length: 11, Branching: True)
- **Turn 1 (Customer 454883)** | `Thu Nov 02 04:57:53`: Video playback stuttering in Final Cut Pro X after update.
- **Turns 2–10**: Apple advises reaching out to FCPX team; customer calls, experiences time zone delays, tests proxy settings, attempts clean reload.
- **Turn 11 (Customer 454883)** | `Wed Nov 08 06:46:53`:  
  > *"@AppleSupport Things have spiraled from worse to 5 levels worse!!! The resolve they had for me was not the problem and rid myself of all 3rd party plugins !! Anyway, have to re-install my system with Sierra if I can ...I’ll stay away from being “High” yep, see what I did there !!"*
- **Order Coherence**: **Yes**. Multi-day saga tracked cleanly across 11 nodes.
- **Resolution Tag Judgment**: **Yes (Accurate)**. Customer is profoundly dissatisfied and problem remains unsolved. Correctly tagged `pending_brand_reply` (`resolved: false`).

---

#### 13. Thread `T_360269` (Length: 8, Branching: True)
- **Turn 1 (Customer 201322)** | `Sun Oct 08 13:57:58`: *"CAN AUTO BRIGHTNESS KINDLY FUCK OFF AND STOP BLINDING ME"*
- **Turn 2 (Customer 201321)** | `Sun Oct 08 13:58:50`: *"@201322 @115858 THIS IS THE WORST FEATURE ON THE NEW UPDATE I NEED TO TURN OFF AUTO BRIGHTNESS ITS CRAP"*
- **Turn 3 (Brand AppleSupport)** | `Sun Oct 08 16:56:00`: *"@201321 In iOS 11, auto-brightness can be controlled via Settings &gt; General &gt; Accessibility &gt; Display Accommodations."*
- **Turn 4 (Customer 201321)** | `Sun Oct 08 16:56:50`: *"@AppleSupport Thank you Apple!!"*
- **Turns 5–8 (Customers 201321 & 201322)** | `16:57 - 17:11`: Friends joking between themselves (*"I'm shook apple replied to us"*, *"IM SCREAMING"*, *"I made it in life"*, *"Honestly same 😂"*).
- **Order Coherence**: **Yes**. Multi-user reply thread maintained accurately.
- **Resolution Tag Judgment**: **FLAGGED: FALSE NEGATIVE (UNRESOLVED TAG ON RESOLVED DIALOG)**.
  - Heuristic Tag: `pending_brand_reply` (`resolved: false`)
  - Human Review: Apple provided the exact settings navigation path, and customer 201321 thanked Apple. However, because the friends continued bantering after the resolution, the final tweet was from a customer without a gratitude keyword, causing the heuristic to classify it as awaiting brand reply.

---

#### 14. Thread `T_582708` (Length: 2, Branching: False)
- **Turn 1 (Customer 257390)** | `Sun Dec 03 05:47:30 +0000 2017`:  
  > *"@AppleSupport This was me when I updated to iOS 11.2 and didn’t have Apple Pay Cash and have to wait https://t.co/GUFbyjlF4F"*
- **Turn 2 (Brand AppleSupport)** | `Sun Dec 03 13:36:29 +0000 2017`:  
  > *"@257390 If you're in the US, those features will require iOS 11.2 and will begin rolling out early next week."*
- **Order Coherence**: **Yes**. Direct answer to customer inquiry.
- **Resolution Tag Judgment**: **Yes (Accurate)**. The tweet occurred only 9.6 hours before the dataset cutoff (Dec 03 23:14:01). The 24-hour dormancy window could not be confirmed. Correctly tagged `brand_final_reply_recent_insufficient_observation_window` (`resolved: unknown`).

---

#### 15. Thread `T_663852` (Length: 3, Branching: False)
- **Turn 1 (Customer 278209)** | `Wed Oct 04 20:56:49 +0000 2017`:  
  > *"I just updated the iOS on my iPad and know I can’t find any of the signs on the keyboard Yo @115858 what’s the dealio"*
- **Turn 2 (Brand AppleSupport)** | `Wed Oct 04 21:22:56 +0000 2017`:  
  > *"@278209 Happy to assist you. DM us what iOS version you're using on the iPad and we'll continue there. https://t.co/GDrqU22YpT"*
- **Turn 3 (Customer 278209)** | `Wed Oct 04 22:02:12 +0000 2017`:  
  > *"@AppleSupport It’s not something you can help with, it’s just how you guys chose to redesign the keyboard"*
- **Order Coherence**: **Yes**. Coherent conversation.
- **Resolution Tag Judgment**: **Yes (Accurate)**. Thread ended with customer expressing dissatisfaction with UI redesign; brand did not reply. Correctly tagged `pending_brand_reply` (`resolved: false`).

---

## Synthesis of Findings & Limitations

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Reconstruction Order Coherence** | 15 / 15 | **100.0%** |
| **Resolution Tag Human Agreement** | 11 / 15 | **73.3%** |
| **False Positives (Gratitude Keyword Ambiguity)** | 3 / 15 | **20.0%** |
| **False Negatives (Customer Chatter Trailing)** | 1 / 15 | **6.7%** |

### Known Heuristic Limitations Documented:
1. **Keyword Polysemy**: Words like `"fixed"` appear in complaints (*"fix this"*, *"if not fixed"*), triggering gratitude rules prematurely.
2. **Sarcasm / Irony**: Colloquial terms like `"cheers"` or `"thanks for nothing"` are treated as gratitude by simple regex dictionaries.
3. **Trailing Social Chatter**: When multiple customers banter after receiving a solution, the thread ends on a non-gratitude customer tweet, misclassifying a resolved thread as `pending_brand_reply`.
4. **DM Black Box**: Dormancy heuristic assumes silence implies resolution, which accurately captures lack of further public complaint, but masks private conversations in DM.
