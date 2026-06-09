# Comparative Study of Rule-Based and AI-Powered Chatbot Systems

**Course Assignment — Conversational Agents**
**Technologies: AIML (python-aiml) and OpenAI GPT API**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Technologies Used](#2-technologies-used)
3. [Implementation Details](#3-implementation-details)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Conclusion and Future Work](#5-conclusion-and-future-work)
6. [References](#6-references)

---

## 1. Introduction

### 1.1 Conversational Agents

Conversational agents — commonly referred to as chatbots — are software systems designed to simulate human-like dialogue. They accept natural language input from a user and produce a relevant, contextually appropriate response. The concept dates back to the 1960s with Joseph Weizenbaum's ELIZA, a program that mimicked a psychotherapist by applying simple pattern-substitution rules to user input (Weizenbaum, 1966). Since then, the field has evolved dramatically, branching into two broad paradigms: rule-based systems that rely on hand-crafted patterns and templates, and machine-learning-based systems that derive conversational behaviour from large corpora of text.

Modern conversational agents span a wide spectrum of complexity. At one end sit simple FAQ bots that match keywords to canned responses; at the other end are large language model (LLM)-powered assistants capable of multi-turn reasoning, code generation, and nuanced emotional understanding. Both ends of this spectrum have practical value, and the choice between them depends heavily on the requirements of the deployment context.

### 1.2 Importance of Chatbots Across Industries

Chatbots have become a significant component of digital infrastructure across many sectors:

- **Customer Service**: Organisations deploy chatbots to handle high volumes of routine enquiries — order tracking, account management, FAQs — reducing operational costs and improving response times (IBM, 2023).
- **Healthcare**: Symptom-checking bots and appointment schedulers help triage patients and reduce administrative burden on clinical staff (Laranjo et al., 2018).
- **Education**: Intelligent tutoring systems and study assistants provide personalised feedback to learners at scale, adapting to individual progress (Winkler & Söllner, 2018).
- **E-Commerce**: Product recommendation bots and conversational search interfaces improve user engagement and conversion rates (Xu et al., 2017).
- **Mental Health**: Chatbots such as Woebot provide cognitive-behavioural therapy (CBT) exercises and emotional support, extending access to mental health resources (Fitzpatrick et al., 2017).

The global chatbot market was valued at approximately USD 5.1 billion in 2022 and is projected to grow at a compound annual growth rate (CAGR) of over 25% through 2030, reflecting the technology's expanding commercial relevance (Grand View Research, 2023).

### 1.3 Project Objective

This project implements and compares two distinct chatbot systems built with fundamentally different underlying technologies:

1. **AIML Chatbot** (`chatbot1-aiml/`) — a rule-based chatbot implemented using Artificial Intelligence Markup Language (AIML) and the `python-aiml` interpreter. It responds to user input by matching against a library of hand-authored patterns and produces templated responses. In-session memory is achieved through AIML's `<set>` and `<get>` predicates.

2. **OpenAI Chatbot** (`chatbot2-openai/`) — an AI-powered chatbot that delegates language understanding and generation to OpenAI's GPT-3.5-turbo model via the OpenAI API. It maintains a rolling conversation history to provide contextual continuity and persists user information across sessions using a JSON-based user profile.

Both chatbots operate as command-line interface (CLI) applications written in Python. The objective is to evaluate each approach across dimensions including ease of development, flexibility, scalability, interaction quality, and memory handling, thereby informing a practical understanding of when each technology is most appropriate.

---

## 2. Technologies Used

### 2.1 AIML — Artificial Intelligence Markup Language

#### Overview

AIML is an XML-based markup language originally developed by Richard Wallace for the ALICE chatbot project in the late 1990s (Wallace, 2009). It defines a set of `<category>` elements, each containing a `<pattern>` (the input to match) and a `<template>` (the response to produce). The AIML interpreter normalises user input to uppercase, strips punctuation, and attempts to find the most specific matching pattern. If no specific pattern matches, a wildcard catch-all (`<pattern>*</pattern>`) provides a fallback response.

Key AIML constructs used in this project include:

- **`<pattern>`** — defines the input pattern, supporting wildcards (`*` matches one or more words).
- **`<template>`** — defines the response, which may include static text, AIML tags, or references to stored variables.
- **`<set name="...">` / `<get name="...">`** — store and retrieve named predicates (variables) within the AIML kernel's memory for the duration of a session.
- **`<star/>`** — inserts the text matched by the wildcard in the pattern into the template.
- **`<condition>`** — enables conditional branching based on the value of a stored predicate.

The Python library `python-aiml` (version 0.8.6) provides an `aiml.Kernel` class that loads `.aiml` files, normalises input, and executes pattern matching. The `kernel.respond(input)` method returns the matched template response as a string.

#### Justification for Selection

AIML was selected for this project because it represents the canonical rule-based chatbot paradigm and is well-documented in academic literature. It requires no external API calls, no internet connectivity, and no API key, making it straightforward to run in any environment. Its deterministic, transparent behaviour makes it easy to reason about and debug. For a university assignment comparing chatbot technologies, AIML provides a clear and instructive contrast to the probabilistic, generative approach of large language models.

### 2.2 OpenAI API

#### Overview

The OpenAI API provides programmatic access to OpenAI's family of large language models, including the GPT (Generative Pre-trained Transformer) series. GPT models are transformer-based neural networks trained on vast corpora of internet text using a combination of unsupervised pre-training and reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022). The result is a model capable of generating fluent, contextually coherent natural language across a wide range of topics and tasks.

This project uses the `gpt-3.5-turbo` model via the Chat Completions endpoint. The API accepts a list of message objects, each with a `role` (`system`, `user`, or `assistant`) and `content` (the message text). The `system` role allows the developer to provide high-level instructions that shape the model's behaviour throughout the conversation. By including the full conversation history in every API request, the model can reference earlier turns and produce contextually aware responses.

The `openai` Python library (version ≥ 1.0) wraps the HTTP API in a convenient client interface:

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=conversation_history
)
```

#### Justification for Selection

The OpenAI API was selected because it represents the current state of the art in accessible, production-grade conversational AI. GPT-3.5-turbo offers a strong balance between capability and cost, making it practical for a student project. Unlike training a model from scratch, the API approach allows developers to leverage a powerful pre-trained model with minimal infrastructure. This makes it an ideal representative of the modern AI-powered chatbot paradigm, providing a meaningful contrast to the rule-based AIML approach.

---

## 3. Implementation Details

### 3.1 AIML Chatbot — Development Process

The AIML chatbot is structured as follows:

```
chatbot1-aiml/
├── bot.py
├── aiml/
│   ├── greetings.aiml
│   ├── memory.aiml
│   └── fallback.aiml
├── requirements.txt
└── README.md
```

Development began with authoring the `.aiml` rule files, which define the chatbot's entire conversational repertoire. Three files were created with distinct responsibilities:

- **`greetings.aiml`** — handles common greeting patterns (`HELLO`, `HI`, `HEY`, `HOW ARE YOU`, time-of-day greetings). Each pattern maps to a friendly, static response.
- **`memory.aiml`** — implements personalisation through `<set>` and `<get>` predicates. Patterns such as `MY NAME IS *`, `MY FAVORITE COLOR IS *`, and `I LIKE *` capture user-provided information and store it in named variables (`username`, `favcolor`, `hobby`). Recall patterns (`WHAT IS MY NAME`, `WHAT DO I LIKE`) retrieve these values and embed them in responses. Conditional logic via `<condition>` handles the case where a variable has not yet been set.
- **`fallback.aiml`** — provides a single catch-all `<pattern>*</pattern>` category that returns a helpful default message when no other pattern matches.

The entry point `bot.py` initialises an `aiml.Kernel` instance, suppresses verbose output, and loads all `.aiml` files from the `aiml/` directory using `glob`. It then enters a `while True` CLI loop that reads user input, skips whitespace-only strings, checks for exit commands, calls `kernel.respond()`, and prints the result.

### 3.2 AIML Chatbot — Memory and Personalization Mechanism

AIML's memory model is predicate-based. The `aiml.Kernel` instance maintains an internal dictionary of named string variables (predicates) that persist for the lifetime of the process. When a pattern containing `<set name="username"><star/></set>` is matched, the wildcard text is stored under the key `username`. Subsequent patterns that include `<get name="username"/>` retrieve this value and interpolate it into the response.

This mechanism is entirely in-memory and session-scoped: predicates are lost when the process exits. There is no built-in persistence to disk. The `<condition>` tag enables the chatbot to behave differently depending on whether a predicate has been set, providing a rudimentary form of state-aware dialogue.

Example interaction demonstrating memory:

```
You: My name is Alice
Bot: Nice to meet you, Alice! I'll remember your name.

You: What is my name?
Bot: Your name is Alice.

You: I like reading
Bot: That's great! I'll remember that you like reading.

You: What do I like?
Bot: You told me that you like reading.
```

### 3.3 OpenAI Chatbot — Development Process

The OpenAI chatbot is structured as follows:

```
chatbot2-openai/
├── bot.py
├── user_profile.json   (created at runtime)
├── requirements.txt
└── README.md
```

Development centred on `bot.py`, which implements four key responsibilities:

1. **API key validation** — reads `OPENAI_API_KEY` from the environment and exits with a descriptive error if absent, ensuring credentials are never hardcoded.
2. **User profile management** — `load_profile()` reads `user_profile.json` on startup, returning an empty profile `{"name": None, "preferences": {}}` if the file is absent or malformed. `save_profile()` serialises the profile back to disk on exit.
3. **System prompt construction** — `build_system_message()` constructs a `system`-role message from the loaded profile, injecting the user's name and known preferences so the model greets returning users appropriately.
4. **Conversation loop** — maintains a `conversation_history` list initialised with the system message. Each turn appends the user message, calls the OpenAI API with the full history, appends the assistant response, and prints it. `openai.APIError` exceptions are caught and reported without crashing the loop.

### 3.4 OpenAI Chatbot — Memory and Personalization Mechanism

The OpenAI chatbot implements two complementary memory mechanisms:

**In-session memory via conversation history**: The `conversation_history` list accumulates every message exchanged during a session. Because the full list is sent to the API on every request, the model has access to all prior context and can reference earlier statements naturally. This is the standard approach for stateful conversations with the Chat Completions API (OpenAI, 2023).

**Cross-session persistence via user profile**: A `user_profile.json` file stores the user's name and a dictionary of preferences. On startup, this file is loaded and its contents are injected into the system prompt, allowing the model to greet returning users by name and recall their preferences from previous sessions. The profile is saved to disk when the session ends.

Example interaction demonstrating cross-session memory:

```
# Session 1
You: My name is Bob
Bot: Nice to meet you, Bob! How can I help you today?
[User exits — profile saved: {"name": "Bob", "preferences": {}}]

# Session 2 (new process)
Bot: [System prompt includes: "The user's name is Bob."]
You: Do you remember me?
Bot: Of course! You're Bob. Welcome back!
```

### 3.5 Challenges Faced and Solutions

**Challenge 1: AIML pattern normalisation**
AIML normalises all input to uppercase before matching, which means patterns must be written in uppercase. This is a non-obvious constraint that caused initial confusion when patterns failed to match. The solution was to ensure all `<pattern>` elements use uppercase text and to test patterns systematically.

**Challenge 2: AIML wildcard ordering**
AIML applies patterns in order of specificity, with more specific patterns taking precedence over wildcards. Placing the catch-all `<pattern>*</pattern>` in a separate file (`fallback.aiml`) and loading files in alphabetical order ensured that specific patterns in `greetings.aiml` and `memory.aiml` were matched before the fallback.

**Challenge 3: AIML predicate default values**
When a predicate has not been set, `kernel.getPredicate()` returns the string `"unknown"` rather than an empty string or `None`. The `<condition>` tag must explicitly check for `value="unknown"` to handle the unset case. This required careful use of conditional templates in `memory.aiml`.

**Challenge 4: OpenAI API error handling**
Network failures and API rate limits can interrupt the conversation loop. Wrapping the API call in a `try/except openai.APIError` block allows the chatbot to report the error and continue accepting input, rather than crashing. This required understanding the exception hierarchy in the `openai` library (version ≥ 1.0), where `APIError` is the base class for all API-related exceptions.

**Challenge 5: Cross-session profile corruption**
If `user_profile.json` is manually edited and becomes invalid JSON, the chatbot must not crash on startup. The `load_profile()` function catches `json.JSONDecodeError`, prints a warning, and returns an empty profile, ensuring graceful degradation.

---

## 4. Comparative Analysis

### 4.1 Ease of Development

**AIML Chatbot**: Getting started with AIML is straightforward — the XML syntax is readable and the pattern-matching model is intuitive. However, scaling the rule base quickly becomes labour-intensive. Every new topic or phrasing variation requires a new `<category>`. Maintaining consistency across hundreds of patterns is error-prone, and debugging unexpected matches requires careful inspection of the loaded rule set. The total development effort for a non-trivial chatbot is high relative to the conversational coverage achieved.

**OpenAI Chatbot**: The OpenAI chatbot requires minimal domain-specific authoring. The developer writes a system prompt and a conversation loop; the model handles all language understanding and generation. The primary development effort lies in prompt engineering — crafting a system message that reliably shapes the model's behaviour — and in implementing robust error handling and state management. For a broad-domain conversational agent, the OpenAI approach requires significantly less effort to achieve high-quality interactions.

### 4.2 Flexibility

**AIML Chatbot**: AIML is highly inflexible with respect to language variation. A pattern like `MY NAME IS *` will not match "I'm called Alice", "People call me Alice", or "Alice is my name". Each variation requires an explicit additional pattern. The chatbot cannot handle paraphrasing, synonyms, or novel phrasings unless they are explicitly anticipated by the rule author. This rigidity makes AIML poorly suited to open-domain conversation.

**OpenAI Chatbot**: GPT-3.5-turbo handles linguistic variation naturally. It understands paraphrases, synonyms, implicit references, and even ambiguous or poorly formed input. The model can engage on virtually any topic without any domain-specific configuration. This flexibility is one of the primary advantages of LLM-based approaches.

### 4.3 Scalability

**AIML Chatbot**: AIML scales poorly with conversational breadth. Each new topic requires manually authored patterns, and the rule base grows linearly with coverage. Large AIML rule sets (thousands of categories) become difficult to maintain and may exhibit unexpected interactions between patterns. The `python-aiml` library loads all rules into memory at startup, which can be slow for very large rule sets.

**OpenAI Chatbot**: The OpenAI chatbot scales well with conversational breadth — the underlying model already covers a vast range of topics. However, it faces a different scalability challenge: the context window. As `conversation_history` grows, the token count of each API request increases, eventually approaching the model's context limit (4,096 tokens for `gpt-3.5-turbo`). Long sessions may require a truncation or summarisation strategy to remain within limits. Additionally, API costs scale with usage, which may be a constraint for high-volume deployments.

### 4.4 User Interaction Quality

**AIML Chatbot**: Interaction quality is high within the chatbot's defined domain but degrades sharply outside it. Responses are consistent and predictable, which can feel robotic. The chatbot cannot engage in free-form conversation, answer factual questions, or handle unexpected topics gracefully beyond the fallback message. Users who deviate from anticipated phrasings will frequently encounter the fallback response, which can be frustrating.

**OpenAI Chatbot**: Interaction quality is generally high across a wide range of topics. The model produces fluent, contextually appropriate responses and can engage in multi-turn reasoning, answer factual questions, and adapt its tone. Responses are non-deterministic — the same input may produce different outputs on different runs — which can be either a feature (variety) or a drawback (unpredictability) depending on the use case.

### 4.5 Memory and Context Handling

**AIML Chatbot**: Memory is limited to named predicates stored in the AIML kernel. Only information explicitly captured by a `<set>` pattern is retained. The chatbot has no awareness of conversational context beyond what is encoded in predicates — it cannot, for example, refer back to a topic discussed several turns ago unless that topic was explicitly stored. Memory is session-scoped and lost on exit.

**OpenAI Chatbot**: Memory is handled at two levels. Within a session, the full conversation history provides rich contextual continuity — the model can reference any prior statement naturally. Across sessions, the JSON user profile provides lightweight persistence of key facts. This two-tier approach is more powerful and flexible than AIML's predicate system, though it requires careful management of the context window for long sessions.

### 4.6 Pros and Cons Summary

| Dimension | AIML Chatbot | OpenAI Chatbot |
|---|---|---|
| **Setup complexity** | Low — no API key, no external service | Medium — requires API key and internet access |
| **Development effort** | High for broad coverage | Low for broad coverage |
| **Response determinism** | Fully deterministic | Non-deterministic |
| **Language flexibility** | Low — exact pattern matching | High — handles paraphrasing naturally |
| **Domain coverage** | Limited to authored rules | Virtually unlimited |
| **In-session memory** | Predicate-based, explicit only | Full conversation history |
| **Cross-session memory** | None (session-scoped only) | JSON user profile |
| **Scalability (breadth)** | Poor — linear rule growth | Excellent |
| **Scalability (volume)** | Excellent — no API costs | Cost-dependent |
| **Transparency** | High — rules are inspectable | Low — model is a black box |
| **Offline capability** | Yes | No |
| **Privacy** | High — no data leaves the machine | Lower — input sent to OpenAI |
| **Maintenance** | High — rules must be updated manually | Low — model handles language variation |
| **Cost** | Free (open-source library) | Pay-per-token API pricing |

---

## 5. Conclusion and Future Work

### 5.1 Key Findings

This project demonstrated that the two chatbot paradigms — rule-based AIML and LLM-powered OpenAI API — represent fundamentally different trade-offs rather than a simple hierarchy of quality.

The AIML chatbot excels in predictability, transparency, and cost. Its behaviour is fully auditable: every possible response is encoded in the `.aiml` files, and there are no surprises. It requires no internet connectivity, incurs no per-request cost, and raises no data privacy concerns. These properties make it well-suited to narrow, well-defined domains where consistent, controlled responses are paramount.

The OpenAI chatbot excels in flexibility, interaction quality, and ease of development for broad-domain applications. It handles linguistic variation naturally, maintains rich conversational context, and can engage on virtually any topic without domain-specific authoring. Its primary drawbacks are cost, dependency on an external service, non-determinism, and reduced transparency.

### 5.2 Which Approach Suits Which Use Case

**AIML is most appropriate when**:
- The conversational domain is narrow and well-defined (e.g., a FAQ bot for a specific product).
- Responses must be strictly controlled and auditable (e.g., regulated industries such as finance or healthcare).
- The deployment environment has no internet access or strict data privacy requirements.
- Cost is a primary constraint and usage volume is high.

**OpenAI API is most appropriate when**:
- The conversational domain is broad or unpredictable (e.g., a general-purpose assistant).
- High interaction quality and natural language understanding are priorities.
- Development speed is important and the team lacks resources to author extensive rule sets.
- Cross-session personalisation and contextual continuity are required.

### 5.3 Future Improvements

Several enhancements could extend the capabilities of both chatbots:

**AIML Chatbot**:
- **Persistent memory**: Implement serialisation of AIML predicates to a JSON file on exit and reload on startup, mirroring the OpenAI chatbot's cross-session persistence.
- **Expanded rule base**: Author additional `.aiml` files covering more topics (weather, time, general knowledge) to increase conversational coverage.
- **AIML 2.0**: Migrate to AIML 2.0, which introduces richer control flow, including `<srai>` (symbolic reduction) for pattern reuse and `<learn>` for dynamic rule addition at runtime.
- **Hybrid approach**: Combine AIML for structured, domain-specific responses with a fallback to an LLM for out-of-domain queries.

**OpenAI Chatbot**:
- **Context window management**: Implement a sliding window or summarisation strategy to handle long sessions that approach the model's token limit.
- **Richer user profile**: Extend the profile schema to capture more nuanced preferences and use structured extraction (e.g., function calling) to reliably detect and store new information from conversation.
- **Model upgrade**: Migrate to `gpt-4o` or a later model for improved reasoning and instruction-following.
- **Streaming responses**: Use the OpenAI streaming API to display responses token-by-token, improving perceived responsiveness for long outputs.
- **Web interface**: Add a simple web front-end (e.g., using Flask or FastAPI) to make the chatbot accessible via a browser.

**Both chatbots**:
- **Evaluation framework**: Implement automated evaluation metrics (e.g., BLEU score for AIML, human preference ratings for OpenAI) to quantitatively measure interaction quality.
- **Logging**: Add structured logging of conversations (with user consent) to support analysis and iterative improvement.

---

## 6. References

Fitzpatrick, K. K., Darcy, A., & Vierhile, M. (2017). Delivering cognitive behavior therapy to young adults with symptoms of depression and anxiety using a fully automated conversational agent (Woebot): A randomized controlled trial. *JMIR Mental Health*, 4(2), e19. https://doi.org/10.2196/mental.7785

Grand View Research. (2023). *Chatbot market size, share & trends analysis report by type, by product, by application, by region, and segment forecasts, 2023–2030*. https://www.grandviewresearch.com/industry-analysis/chatbot-market

IBM. (2023). *What is a chatbot?* IBM Think. https://www.ibm.com/topics/chatbots

Laranjo, L., Dunn, A. G., Tong, H. L., Kocaballi, A. B., Chen, J., Bashir, R., Surian, D., Gallego, B., Magrabi, F., Lau, A. Y. S., & Coiera, E. (2018). Conversational agents in healthcare: A systematic review. *Journal of the American Medical Informatics Association*, 25(9), 1248–1258. https://doi.org/10.1093/jamia/ocy072

OpenAI. (2023). *Chat completions — OpenAI API documentation*. https://platform.openai.com/docs/guides/chat

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback. *arXiv preprint arXiv:2203.02155*. https://arxiv.org/abs/2203.02155

Wallace, R. S. (2009). The anatomy of ALICE. In R. Epstein, G. Roberts, & G. Beber (Eds.), *Parsing the Turing test: Philosophical and methodological issues in the quest for the talking computer* (pp. 181–210). Springer. https://doi.org/10.1007/978-1-4020-6710-5_13

Weizenbaum, J. (1966). ELIZA — A computer program for the study of natural language communication between man and machine. *Communications of the ACM*, 9(1), 36–45. https://doi.org/10.1145/365153.365168

Winkler, R., & Söllner, M. (2018). Unleashing the potential of chatbots in education: A state-of-the-art analysis. *Academy of Management Annual Meeting Proceedings*, 2018(1). https://doi.org/10.5465/AMBPP.2018.15903abstract

Xu, A., Liu, Z., Guo, Y., Sinha, V., & Akkiraju, R. (2017). A new chatbot for customer service on social media. In *Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems* (pp. 3506–3510). ACM. https://doi.org/10.1145/3025453.3025496
