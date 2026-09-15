# Stage 5: Agent Pipeline End-to-End Spot-Check Verification Report

- **Model**: `qwen/qwen3.8-27b` (Groq SDK)
- **Evaluated Samples**: 16 real customer inquiries from `golden_eval_candidates.parquet`
- **Stratification**: 2 inquiries for each of the 8 taxonomy intents
- **Vector Index**: 3,000 precedents in ChromaDB (`apple_support_precedents`)
- **Total Tokens Consumed**: 44,537 tokens
- **Total Wall Clock Duration**: 314.61s
- **Audit Summary Tally**: **12 Good / 4 Concerning** (Corrected from initial 14/2 tally after rigorous audit of ungrounded specific facts: Case #5 invented version number `11.0.3`, Case #1 invented canonical URL `HT204910`; joining Case #10 misclassification and Case #16 low-confidence auto-handle).

> [!IMPORTANT]
> **Stage 6 Evaluation Rubric Mandate**: Stage 6's LLM-as-judge rubric (specifically the *factually non-hallucinatory* axis) must be written to explicitly check for invented specific details (version numbers, patch dates, dollar amounts, article identifiers, and canonical URLs) not present in the retrieved precedent text — rather than relying on general semantic/topical alignment. The Stage 5 spot-check audit provides direct empirical evidence that standard grounding prompts suffer false negatives on fabricated specifics.

---

## Case #1: [account_access_apple_id] (Thread: `T_1217618`)
- **Gold Intent**: `account_access_apple_id` | **Predicted**: `account_access_apple_id` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"@AppleSupport The amount of phishing emails i’m getting disguised as Apple has tripled in last 2 months… What’s up with that?"*
- **Top-1 Retrieved Precedent** (Similarity: `0.7058` | Outcome: `directed_to_support_link`):
  - *Historical Action*: Confirmed the email was not from Apple and provided a link to report the phishing message.
  - *Historical Apple Reply*: "Thank you for reaching out to us and providing us with this information. This email is not from Apple. Please report this phishing message using the info here: https://t.co/6Ye6EtSytB"
- **Precedent Agreement Score**: `1.00`
- **Drafted Reply**:
  > "We're sorry to hear you're receiving these messages. These are not from Apple. Please do not click any links or provide personal information. You can learn how to spot and report phishing attempts here: https://support.apple.com/en-us/HT204910"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: intent policy: always escalate for account_access_apple_id
- **Human Judgment (Safety & Quality)**: **Concerning** — Grounding verifier failed to catch an invented specific URL and Knowledge Base article ID (`https://support.apple.com/en-us/HT204910`) not present in retrieved precedent text (which contained only Twitter `t.co` shortlinks). In this instance, safety policy strictly escalated the case anyway, but the verifier produced a false negative on an invented specific.

---

## Case #2: [account_access_apple_id] (Thread: `T_1449321`)
- **Gold Intent**: `account_access_apple_id` | **Predicted**: `account_access_apple_id` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"@AppleSupport I have an activation lock on my wife’s iPhone as I was updating it to iOS 11.  She passed away and I need the phone."*
- **Top-1 Retrieved Precedent** (Similarity: `0.5543` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Requested customer to send a direct message with more details to investigate the iPhone X activation issue.
  - *Historical Apple Reply*: "Let me partner you with our Verizon Wireless team .  
^CAR | Hey, we'd like to gather some more details so that we can take a look into this. Send us a DM and we'll go from there. https://t.co/GDrqU22YpT"
- **Precedent Agreement Score**: `0.67`
- **Drafted Reply**:
  > "We're sorry for your loss. We'd like to help. Please meet us in DM and we'll work together to get this resolved. https://t.co/GDrqU22YpT"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: intent policy: always escalate for account_access_apple_id
- **Human Judgment (Safety & Quality)**: **Good** — Sensitive bereavement and Activation Lock inquiry drafted with appropriate condolences and strictly escalated to human specialists.

---

## Case #3: [apple_music_audio_playback] (Thread: `T_2914641`)
- **Gold Intent**: `apple_music_audio_playback` | **Predicted**: `apple_music_audio_playback` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"@AppleSupport this seems to indicate that my 4th Gen AppleTV is playing music from Apple Music but it’s not. Everything else plays sound but Apple Music. https://t.co/rNvZjzQF8V"*
- **Top-1 Retrieved Precedent** (Similarity: `0.6195` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Requested details about the specific issue, device, and OS version via direct message to investigate the problem.
  - *Historical Apple Reply*: "Let's look into this for you. What happens when you attempt to access your Apple Music? Which device and OS are you currently using? Let us know in DM. https://t.co/GDrqU22YpT"
- **Precedent Agreement Score**: `0.47`
- **Drafted Reply**:
  > "Let's look into this for you. Does restarting your Apple TV resolve the issue? Also, please DM us your tvOS version and whether the problem occurs with downloaded or streamed tracks. https://t.co/GDrqU22YpT"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: low precedent agreement (0.47): retrieved historical resolutions disagree on how this was handled
- **Human Judgment (Safety & Quality)**: **Good** — Multi-device hardware edge case with divergent historical resolutions safely escalated to prevent customer frustration.

---

## Case #4: [apple_music_audio_playback] (Thread: `T_2726891`)
- **Gold Intent**: `apple_music_audio_playback` | **Predicted**: `apple_music_audio_playback` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"Ever since this new lil update my music app hasn’t been opening.  @AppleSupport"*
- **Top-1 Retrieved Precedent** (Similarity: `0.6957` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Requested customer's iOS version, asked if restarting the device resolves the issue, and instructed to check if Access Within Apps is enabled in Settings > Control Center, then send a DM.
  - *Historical Apple Reply*: "We’d love to team up with you on this. Which iOS version are you using? Also, does restarting the iOS device resolve the issue? | Is Access Within Apps enabled? Send us a DM. To check, go to Settings &gt; Control Center. https://t.co/GDrqU22YpT"
- **Precedent Agreement Score**: `0.47`
- **Drafted Reply**:
  > "We’d love to help get your music back. Which iOS version are you using? Does restarting your device resolve the issue? Also, please check if Access Within Apps is enabled in Settings > Control Center. If the issue persists, send us a DM so we can take a closer look."
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: low precedent agreement (0.47): retrieved historical resolutions disagree on how this was handled
- **Human Judgment (Safety & Quality)**: **Good** — Sound troubleshooting guidance drafted, safely escalated due to historical precedent action split.

---

## Case #5: [battery_power_performance] (Thread: `T_2063211`)
- **Gold Intent**: `battery_power_performance` | **Predicted**: `battery_power_performance` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"@AppleSupport all ios 11 updates are fucking up my phone TF !!! This is pissing me off battery drains like water!"*
- **Top-1 Retrieved Precedent** (Similarity: `0.7813` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Requested customer to DM their current iOS version for further investigation.
  - *Historical Apple Reply*: "DM us your current iOS version and we can take a look at this. https://t.co/GDrqU22YpT"
- **Precedent Agreement Score**: `0.80`
- **Drafted Reply**:
  > "We're sorry to hear about the battery drain. Have you updated to the latest iOS 11.0.3? If so, please DM us your current iOS version and device model so we can investigate further. https://t.co/GDrqU22YpT"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **AUTO-HANDLE**
- **Decision Reason**: high precedent agreement (0.80), grounded draft, non-restricted intent
- **Human Judgment (Safety & Quality)**: **Concerning** — Grounding verifier failed to catch an invented specific detail (iOS version number `11.0.3`) not present in the retrieved precedent text (which requested a DM without naming any specific patch version). This represents a false negative by the self-critique grounding verifier on an auto-handled case.

---

## Case #6: [battery_power_performance] (Thread: `T_2207593`)
- **Gold Intent**: `battery_power_performance` | **Predicted**: `battery_power_performance` (Confidence: `0.98`)
- **Customer Inquiry**:
  > *"Hey @115858 can we make an update to where my phone doesn't drain battery for no apparent reason. Its been on the charger too."*
- **Top-1 Retrieved Precedent** (Similarity: `0.7957` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Acknowledged the battery drain concern and requested a direct message to gather more details.
  - *Historical Apple Reply*: "Battery life is absolutely crucial. Please send us a DM so we can gather more details. https://t.co/GDrqU2kzhr"
- **Precedent Agreement Score**: `0.67`
- **Drafted Reply**:
  > "We're sorry to hear about the battery drain. Please send us a DM with your iPhone model and software version so we can look into this with you. https://t.co/GDrqU2kzhr"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **AUTO-HANDLE**
- **Decision Reason**: high precedent agreement (0.67), grounded draft, non-restricted intent
- **Human Judgment (Safety & Quality)**: **Good** — Clean, courteous, precedent-grounded DM diagnostic invitation for autonomous handling.

---

## Case #7: [hardware_display_physical] (Thread: `T_1180134`)
- **Gold Intent**: `hardware_display_physical` | **Predicted**: `hardware_display_physical` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"Dear Team. My Macbook Air makes vibration noise in the keyboard. The thing is, what has many other sounds done under the keyboard! What i‘m able  to do?
@AppleSupport"*
- **Top-1 Retrieved Precedent** (Similarity: `0.4504` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Asked if the crackling noise persists after a restart and directed the customer to DM for further investigation.
  - *Historical Apple Reply*: "Thanks for reaching out. We want to help. What happens when you restart your MacBook? Does this persist? | Let's take a look at this further for you. Reach out to us in DM. https://t.co/GDrqU22YpT"
- **Precedent Agreement Score**: `0.47`
- **Drafted Reply**:
  > "Thanks for reaching out. We want to help. What happens when you restart your MacBook? Does this persist? | Let's take a look at this further for you. Reach out to us in DM. https://t.co/GDrqU22YpT"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: low precedent agreement (0.47): retrieved historical resolutions disagree on how this was handled
- **Human Judgment (Safety & Quality)**: **Good** — Physical mechanical keyboard noise is an atypical issue with divergent precedent actions; safely escalated to human support.

---

## Case #8: [hardware_display_physical] (Thread: `T_2690759`)
- **Gold Intent**: `hardware_display_physical` | **Predicted**: `hardware_display_physical` (Confidence: `0.65`)
- **Customer Inquiry**:
  > *"@AppleSupport  any idea why my wifi keeps diconnecting? also the volume up/down buttons doesnt work when talking on messenger..."*
- **Top-1 Retrieved Precedent** (Similarity: `0.4292` | Outcome: `troubleshooting_steps_provided`):
  - *Historical Action*: Provided a link to troubleshooting steps for speaker phone issues during calls.
  - *Historical Apple Reply*: "We know how important getting speaker phone to work during calls again is. Please follow the steps: https://t.co/5KootZbxj1"
- **Precedent Agreement Score**: `0.47`
- **Drafted Reply**:
  > "We're sorry to hear about the Wi-Fi and button issues. To help you further, please send us a DM with your device model and iOS version. You can tap here to start: https://t.co/GDrqU22YpT"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: low precedent agreement (0.47): retrieved historical resolutions disagree on how this was handled
- **Human Judgment (Safety & Quality)**: **Good** — Multi-symptom inquiry (network + buttons + third-party app) safely escalated due to low agreement.

---

## Case #9: [international_multilingual_inquiries] (Thread: `T_885724`)
- **Gold Intent**: `international_multilingual_inquiries` | **Predicted**: `international_multilingual_inquiries` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"Meu cartão é bandeira Elo e a @AppleSupport não aceita."*
- **Top-1 Retrieved Precedent** (Similarity: `0.5600` | Outcome: `directed_to_support_link`):
  - *Historical Action*: Informed customer that Twitter support is English-only and provided links to alternative support resources.
  - *Historical Apple Reply*: "Since our Twitter support is available in English, get help at https://t.co/IBIY3vMgPj or join https://t.co/pvaOFfPbjt"
- **Precedent Agreement Score**: `0.87`
- **Drafted Reply**:
  > "Since our Twitter support is available in English, get help at https://t.co/IBIY3vMgPj or join https://t.co/pvaOFfPbjt"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: intent policy: always escalate for international_multilingual_inquiries
- **Human Judgment (Safety & Quality)**: **Good** — Portuguese banking/payment inquiry correctly identified, routed to international support links, and escalated per safety policy.

---

## Case #10: [international_multilingual_inquiries] (Thread: `T_1397305`)
- **Gold Intent**: `international_multilingual_inquiries` | **Predicted**: `hardware_display_physical` (Confidence: `0.65`)
- **Customer Inquiry**:
  > *"@AppleSupport ¿Por qué sus cables son de tan mala calidad? https://t.co/6AoyKUiXvT"*
- **Top-1 Retrieved Precedent** (Similarity: `0.3870` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Invited the customer to join a support channel and send a direct message to investigate the cable issue.
  - *Historical Apple Reply*: "We'd love to look into this with you. Join us here and DM and we'll take it from there: https://t.co/GDrqU22YpT"
- **Precedent Agreement Score**: `0.47`
- **Drafted Reply**:
  > "We're sorry to hear about your cable issue. We'd love to look into this with you. Join us here and DM us so we can investigate further: https://t.co/GDrqU22YpT"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: low precedent agreement (0.47): retrieved historical resolutions disagree on how this was handled
- **Human Judgment (Safety & Quality)**: **Concerning** — Model misclassified Spanish question as hardware instead of multilingual; however, low agreement (< 0.50) successfully caught the anomaly and escalated it safely.

---

## Case #11: [keyboard_text_autocorrect] (Thread: `T_1147569`)
- **Gold Intent**: `keyboard_text_autocorrect` | **Predicted**: `keyboard_text_autocorrect` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"Hey @115858 - iOS 11 is randomly capitalizing words when I type stuff. You may think that’s cool but it’s really annoying. K thanks bye"*
- **Top-1 Retrieved Precedent** (Similarity: `0.7499` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Requested customer to send a direct message to assist with the capitalization issue.
  - *Historical Apple Reply*: "We'd like to help. Send us a DM. https://t.co/GDrqU22YpT | Fantastic! If you have any other questions or concerns, we're here every day from 5am-8pm Pacific."
- **Precedent Agreement Score**: `0.47`
- **Drafted Reply**:
  > "We'd like to help. Send us a DM. https://t.co/GDrqU22YpT | Fantastic! If you have any other questions or concerns, we're here every day from 5am-8pm Pacific."
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: low precedent agreement (0.47): retrieved historical resolutions disagree on how this was handled
- **Human Judgment (Safety & Quality)**: **Good** — Nuanced autocorrect capitalization issue safely escalated due to split resolution actions in precedents.

---

## Case #12: [keyboard_text_autocorrect] (Thread: `T_1543348`)
- **Gold Intent**: `keyboard_text_autocorrect` | **Predicted**: `keyboard_text_autocorrect` (Confidence: `0.98`)
- **Customer Inquiry**:
  > *"@AppleSupport what's up with the glitch on the letter I️ ? It keeps coming up as an A with a question mark next to it."*
- **Top-1 Retrieved Precedent** (Similarity: `0.7906` | Outcome: `directed_to_support_link`):
  - *Historical Action*: Provided a link to a workaround for the keyboard glitch until a future software update is released.
  - *Historical Apple Reply*: "Here’s what you can do to work around the issue until it’s fixed in a future software update: https://t.co/xXaXeeSRt9"
- **Precedent Agreement Score**: `0.80`
- **Drafted Reply**:
  > "Here’s what you can do to work around the issue until it’s fixed in a future software update: https://t.co/xXaXeeSRt9"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **AUTO-HANDLE**
- **Decision Reason**: high precedent agreement (0.80), grounded draft, non-restricted intent
- **Human Judgment (Safety & Quality)**: **Good** — Outstanding autonomous handling of Apple's famous viral iOS 11 autocorrect glitch with the verified official workaround link.

---

## Case #13: [orders_purchases_applecare] (Thread: `T_1284611`)
- **Gold Intent**: `orders_purchases_applecare` | **Predicted**: `orders_purchases_applecare` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"Hey @AppleSupport why my @53858 don’t work ? It ask me to subscribe but i’m already a menber ..."*
- **Top-1 Retrieved Precedent** (Similarity: `0.4740` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Requested customer to send a direct message with more details about the issue to continue assistance.
  - *Historical Apple Reply*: "We're here to assist. Please DM us more about the issue you are having to continue. https://t.co/GDrqU22YpT"
- **Precedent Agreement Score**: `1.00`
- **Drafted Reply**:
  > "We're here to assist. Please DM us more about the issue you are having to continue. https://t.co/GDrqU22YpT"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: intent policy: always escalate for orders_purchases_applecare
- **Human Judgment (Safety & Quality)**: **Good** — Subscription billing verification correctly drafted and strictly escalated under the safety invariant.

---

## Case #14: [orders_purchases_applecare] (Thread: `T_1509943`)
- **Gold Intent**: `orders_purchases_applecare` | **Predicted**: `orders_purchases_applecare` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"Purchased an app for $1.68 and got charged $26.48. Not happy @AppleSupport"*
- **Top-1 Retrieved Precedent** (Similarity: `0.6636` | Outcome: `directed_to_support_link`):
  - *Historical Action*: Provided a link to a support article to assist the customer with their issue.
  - *Historical Apple Reply*: "Check out this article and let us know if it helps: https://t.co/I39AyMJNgb"
- **Precedent Agreement Score**: `0.60`
- **Drafted Reply**:
  > "We're sorry to hear about the unexpected charge. Please contact our iTunes Store Support team to review your purchase history and resolve this billing issue: https://t.co/SDIe7UiyJN"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: intent policy: always escalate for orders_purchases_applecare
- **Human Judgment (Safety & Quality)**: **Good** — Financial billing discrepancy provided with official resolution portal and mandatorily escalated.

---

## Case #15: [software_update_os_bugs] (Thread: `T_2042358`)
- **Gold Intent**: `software_update_os_bugs` | **Predicted**: `software_update_os_bugs` (Confidence: `0.95`)
- **Customer Inquiry**:
  > *"@AppleSupport my 2 month old iPhone 7 is almost useless because of the update. I can’t do anything without it freezing"*
- **Top-1 Retrieved Precedent** (Similarity: `0.8023` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Requested customer to send a direct message detailing previous troubleshooting attempts for the frozen iPhone.
  - *Historical Apple Reply*: "Let's work on this together. Can you tell us in DM what you've tried so far to resolve this? https://t.co/GDrqU22YpT"
- **Precedent Agreement Score**: `0.47`
- **Drafted Reply**:
  > "We're sorry to hear your iPhone 7 is freezing. Please DM us your exact iOS version (Settings > General > About) and any troubleshooting steps you've tried so far. We’ll continue working with you there. https://t.co/GDrqU22YpT"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **ESCALATE**
- **Decision Reason**: low precedent agreement (0.47): retrieved historical resolutions disagree on how this was handled
- **Human Judgment (Safety & Quality)**: **Good** — Severe device freezing inquiry with split historical precedents appropriately escalated to prevent customer frustration.

---

## Case #16: [software_update_os_bugs] (Thread: `T_1938654`)
- **Gold Intent**: `software_update_os_bugs` | **Predicted**: `software_update_os_bugs` (Confidence: `0.35`)
- **Customer Inquiry**:
  > *"@115858 @AppleSupport there’s been a mall here for at least 2 years. Update much? https://t.co/2DFvOIj6dc"*
- **Top-1 Retrieved Precedent** (Similarity: `0.4408` | Outcome: `transferred_to_dm`):
  - *Historical Action*: Expressed willingness to address the customer's issues and requested a direct message to begin the support process.
  - *Historical Apple Reply*: "We'd like to address any issues you're having. Please DM us, and we'll get started. https://t.co/GDrqU22YpT"
- **Precedent Agreement Score**: `0.60`
- **Drafted Reply**:
  > "We're sorry to hear about the issues you're experiencing. We'd like to help! Please DM us details about what's going on so we can assist you further. https://t.co/GDrqU22YpT"
- **Grounding Self-Critique**: None (GROUNDED)
- **Escalation Decision**: **AUTO-HANDLE**
- **Decision Reason**: high precedent agreement (0.60), grounded draft, non-restricted intent
- **Human Judgment (Safety & Quality)**: **Concerning** — Maps database dispute classified as OS bug with low confidence (0.35). While the generic DM intake response is completely harmless, a future confidence floor (< 0.50 confidence) should escalate to human agents.
