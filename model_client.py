import argparse

from openai import OpenAI

DEFAULT_QUESTION = """Subject: Package Delivered to Neighbor - Building 4, Apt 2B

Tracking: USPS-9405511899562871456789

Delivery Notice

Dear Customer,

Your package was delivered to your neighbor as you were not home.

Delivered To:
- Name: Sarah Johnson
- Address: Building 4, Apartment 2B
- Relationship: Neighbor (same floor)
- Signed: Yes, at 3:15 PM today

Your Address:
Building 4, Apartment 2A
Chicago, IL 60614

Package Info:
- From: Target.com
- Weight: 5.2 lbs
- Service: USPS Priority Mail
- Tracking: 9405511899562871456789

Delivery Attempt:
We attempted delivery at 3:10 PM. No answer at your door. Your neighbor (Apt 2B) answered and accepted the package on your behalf.

Delivery photo: https://usps.com/photo/9405511899562871456789
(Shows your neighbor receiving package)

Pickup Instructions:
Please retrieve your package from Sarah Johnson at Apartment 2B.

Note: USPS policy allows delivery to neighbors in multi-unit buildings when recipient is unavailable and neighbor is willing to accept.

We left a notice on your door.

Questions?
Call: 1-800-ASK-USPS

USPS Delivery Services"""


class DistilLabsLLM(object):
    def __init__(self, model_name: str, api_key: str = "EMPTY", port: int = 11434):
        self.model_name = model_name
        self.client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=api_key)

    def get_prompt(
        self,
        question: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": """
You are a classifier working on a problem described in task_description XML block:
<task_description>## Task
Classify incoming emails into one of ten predefined categories to enable intelligent email organization, prioritization, and automation workflows. The classification must accurately determine the email's primary purpose and intent based on comprehensive analysis of sender information, subject line, body content, formatting patterns, and contextual signals. The system should handle multi-lingual emails (English and French), mixed personal/professional contexts, and edge cases where emails may contain elements of multiple categories.

## Inputs
Complete email content including:
- Subject line (required)
- Email body text (required)
- Sender information (when available)
- Metadata such as timestamps, domains, and formatting (when available)

## Outputs
A single category label that best represents the email's primary purpose and expected user action.

## Classification Guidelines
1. **Multi-category Resolution:** When an email contains elements of multiple categories, classify based on the PRIMARY user action required. Priority order for ambiguous cases: Security > Spam > Billing > Work > Travel > Shipping > Personal > Promotional > Newsletter > Other.
2. **Language Handling:** The system must accurately classify emails in both English and French based on content meaning. French keywords (e.g., 'facture', 'virement', 'livraison') must be recognized.
3. **Context Awareness:** Consider sender domain and structure. E.g., '@linkedin.com' about jobs is AI/Work, but about network posts is AI/Newsletter.
4. **Edge Case Principles:**
   - Security concerns always take precedence.
   - Obvious spam/phishing is always AI/Spam.
   - Transactional emails (receipts) go to AI/Billing.
   - Personal relationships override platform context.
   - Work context is determined by professional intent, not just sender.

## Decision Logic Examples
- **LinkedIn Flow:** Job posting = AI/Work; Personal msg = AI/Personal; Digest = AI/Newsletter; Profile view = AI/Other.
- **Payment Flow:** If amount+ID present = AI/Billing; If phishing/scam = AI/Spam; If shipping focus = AI/Shipping.
- **Notification Flow:** Security alert = AI/Security; Payment = AI/Billing; Delivery = AI/Shipping; Personal msg = AI/Personal.</task_description>
Classify the input into one of the available classes, each class has a name in class_name and description in class_description XML block:

<class_name>AI/Promotional</class_name>
<class_description>Marketing and sales communications from businesses, services, or platforms promoting products, services, features, or special offers. INDICATORS: Discount codes, limited-time deals, product launches, 'Upgrade today' calls-to-action, webinar invitations. EXAMPLES: SaaS discount offers, early access invites, Black Friday sales. EDGE CASES: Work-related webinar invites from vendors count as Promotional.</class_description>


<class_name>AI/Travel</class_name>
<class_description>All communications related to travel arrangements, transportation bookings, accommodations, and trip logistics. INDICATORS: Flight/Hotel confirmations, boarding passes, car rental reservations, itineraries. EXAMPLES: Air France confirmations, Airbnb bookings, Eurostar tickets. EDGE CASES: Work conference travel is AI/Travel (logistics focus), not AI/Work.</class_description>


<class_name>AI/Spam</class_name>
<class_description>Unsolicited, fraudulent, or malicious emails including phishing attempts, scams, lottery notifications, and suspicious requests. INDICATORS: Unrealistic promises ('You won!'), urgent threats, requests for passwords/SSN, generic greetings, poor grammar, mismatched sender domains. EXAMPLES: Phishing impersonating Amazon/banks, inheritance scams, crypto schemes. EDGE CASES: Legitimate security alerts go to AI/Security; aggressive but legitimate marketing goes to AI/Promotional.</class_description>


<class_name>AI/Shipping</class_name>
<class_description>Order fulfillment communications including shipping confirmations, tracking updates, delivery notifications, and returns. INDICATORS: Tracking numbers (UPS/FedEx), 'Out for delivery' status, delivered photos, return labels. EXAMPLES: Amazon shipment updates, UPS delivery notifications. EDGE CASES: Order confirmations without shipping info often go to AI/Billing.</class_description>


<class_name>AI/Security</class_name>
<class_description>Account security notifications including login alerts, authentication codes, and password changes. INDICATORS: New device logins, 2FA codes, password reset requests, suspicious activity alerts. EXAMPLES: Google sign-in alerts, Microsoft 2FA codes, GitHub security keys. EDGE CASES: If the email is a scam threat, it is AI/Spam.</class_description>


<class_name>AI/Billing</class_name>
<class_description>Financial transaction communications including invoices, payment receipts, subscription charges, and refunds. INDICATORS: Invoice numbers, transaction IDs, 'Payment successful', subscription renewals, tax receipts. EXAMPLES: Stripe receipts, Netflix renewals, cloud billing statements. EDGE CASES: Order confirmations with payment info are AI/Billing; Expired trial upsells are AI/Promotional.</class_description>


<class_name>AI/Work</class_name>
<class_description>Professional and employment-related communications including job opportunities, project updates, team collaboration, and career development. INDICATORS: Job postings, meeting agendas, sprint planning, pull requests, performance reviews, client emails. EXAMPLES: LinkedIn Recruiter messages, Jira updates, Slack digest, client project requirements. EDGE CASES: Professional conference travel is AI/Travel; Work-related SaaS receipts are AI/Billing.</class_description>


<class_name>AI/Newsletter</class_name>
<class_description>Curated content digests, regular informational updates, or periodic communications from subscribed sources. INDICATORS: Daily/weekly cadence, multiple article links, 'digest', 'roundup', unsubscribe links. EXAMPLES: TechCrunch daily, GitHub trending, Substack newsletters. EDGE CASES: A single dedicated promotional email from a newsletter sender is AI/Promotional.</class_description>


<class_name>AI/Personal</class_name>
<class_description>Direct personal communications from friends, family, or colleagues regarding non-professional matters. INDICATORS: Casual tone, social plans (coffee/dinner), birthday wishes, personal advice. EXAMPLES: Friend asking to hang out, family updates, personal networking. EDGE CASES: Colleagues emailing about work are AI/Work; Platform notifications about messages are AI/Other.</class_description>


<class_name>AI/Other</class_name>
<class_description>Miscellaneous communications including platform notifications, system messages, event registrations, and administrative notices. INDICATORS: Automated system updates, terms of service changes, community moderation, badge awards, meetup confirmations. EXAMPLES: Reddit upvote notifications, Terms of Service updates, event registrations. EDGE CASES: Security notifications must go to AI/Security.</class_description>

Write the name of the predicted class inside output XML block
For example, if the input matches class test_output, write
<output>test_output</output>
""",
            },
            {
                "role": "user",
                "content": f"""

Now for the real task, classify the following example
<question>{question}</question>
""",
            },
        ]

    def invoke(self, question: str) -> str:
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.get_prompt(question),
            temperature=0,
            reasoning_effort="none",
        )
        return chat_response.choices[0].message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION, required=False)
    parser.add_argument("--api-key", type=str, default="EMPTY", required=False)
    parser.add_argument("--model", type=str, default="model", required=False)
    parser.add_argument("--port", type=int, default=11434, required=False)
    args = parser.parse_args()

    client = DistilLabsLLM(model_name=args.model, api_key=args.api_key, port=args.port)

    print(client.invoke(args.question))