How to label your emails locally with a distil labs fine-tuned model and n8n 

We built a fully local Gmail auto-labeler with n8n + a fine-tuned 0.6B model (no email content sent to cloud LLMs)

Most of our inboxes are a mix of useful and distracting. Labels can help making order from the chaos - but labelling all emails manually takes time too.

We put together a setup that auto-labels Gmail **locally**, so email content does not go to external LLM APIs.

What it does (end-to-end local):

- n8n trigger when you receive an email
- It sends the email text (subject + snippet/body) to a fine-tuned model running on localhost via Ollama
- It applies the predicted label back in Gmail (we recommend prefixing labels with AI/)

Label set (10-way closed set):
Billing, Newsletter, Work, Personal, Promotional, Security, Shipping, Travel, Spam, Other

Results:
