---
title:  "Building Agents"
description: >-
  Building agents.
author: lso
date:   2025-02-14 11:08:03 +0200
categories: [Blogging, Tutorial]
tags: [agents,llm]
pin: false
media_subpath: '/posts/20250214'
---

## What is an agent?

An agent is an AI model capable of reasoning, planning, and interacting with its environment. We call it Agent because it has agency, aka it has the ability to interact with the environment.

The main steps of an agent are:
* Reasoning
* Planning
* Action Execution by using tools


## Agent's brain - the LLM

Most LLMs nowadays are built on the Transformer architecture—a deep learning architecture based on the “Attention” algorithm, that has gained significant interest since the release of BERT from Google in 2018.

There are 3 types of transformers :

1. Encoder: An encoder-based Transformer takes text (or other data) as input and outputs a dense representation (or embedding) of that text. Use Cases: Text classification, semantic search, Named Entity Recognition.
2. Decoder: A decoder-based Transformer focuses on generating new tokens to complete a sequence, one token at a time. Use Cases: Text generation, chatbots, code generation
3. Seq2Seq (Encoder–Decoder): A sequence-to-sequence Transformer combines an encoder and a decoder. The encoder first processes the input sequence into a context representation, then the decoder generates an output sequence. Use Cases: Translation, Summarization, Paraphrasing.

Although Large Language Models come in various forms, LLMs are typically decoder-based models with billions of parameters. 

The underlying principle of an LLM is simple yet highly effective: its objective is to predict the next token, given a sequence of previous tokens. A “token” is the unit of information an LLM works with. You can think of a “token” as if it was a “word”, but for efficiency reasons LLMs don’t use whole words.

For example, while English has an estimated 600,000 words, an LLM might have a vocabulary of around 32,000 tokens (as is the case with Llama 2). Tokenization often works on sub-word units that can be combined.

Each LLM has some special tokens specific to the model. The LLM uses these tokens to open and close the structured components of its generation. For example, to indicate the start or end of a sequence, message, or response. Moreover, the input prompts that we pass to the model are also structured with special tokens. The most important of those is the End of sequence token (EOS).

LLMs are said to be autoregressive, meaning that the output from one pass becomes the input for the next one. This loop continues until the model predicts the next token to be the EOS token, at which point the model can stop.

### System Messages

System messages (also called System Prompts) define how the model should behave. They serve as persistent instructions, guiding every subsequent interaction.

```
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

When using Agents, the System Message also gives information about the available tools, provides instructions to the model on how to format the actions to take, and includes guidelines on how the thought process should be segmented.

### Conversations: User and Assistant Messages

A conversation consists of alternating messages between a Human (user) and an LLM (assistant).

Chat templates help maintain context by preserving conversation history, storing previous exchanges between the user and the assistant. This leads to more coherent multi-turn conversations.

```
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```
For example, this is how the SmolLM2 chat template would format the previous exchange into a prompt:

```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
I need help with my order<|im_end|>
<|im_start|>assistant
I'd be happy to help. Could you provide your order number?<|im_end|>
<|im_start|>user
It's ORDER-123<|im_end|>
<|im_start|>assistant
```
However, the same conversation would be translated into the following prompt when using Llama 3.2:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 10 Feb 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

I need help with my order<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'd be happy to help. Could you provide your order number?<|eot_id|><|start_header_id|>user<|end_header_id|>

It's ORDER-123<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Chat-Templates

Chat templates are essential for structuring conversations between language models and users. They guide how message exchanges are formatted into a single prompt.

#### Base Models vs. Instruct Models

A Base Model is trained on raw text data to predict the next token.

An Instruct Model is fine-tuned specifically to follow instructions and engage in conversations. For example, SmolLM2-135M is a base model, while SmolLM2-135M-Instruct is its instruction-tuned variant.

To make a Base Model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in.

To make a Base Model behave like an instruct model, we need to format our prompts in a consistent way that the model can understand. This is where chat templates come in.

It’s important to note that a base model could be fine-tuned on different chat templates, so when we’re using an instruct model we need to make sure we’re using the correct chat template.

Because each instruct model uses different conversation formats and special tokens, chat templates are implemented to ensure that we correctly format the prompt the way each model expects.

Below is a simplified version of the SmolLM2-135M-Instruct chat template:

```
{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face
<|im_end|>
{% endif %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
```

Given these messages:

```
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."},
    {"role": "user", "content": "How do I use it ?"},
]
```

The chat template will format them as follows:

```
<|im_start|>system
You are a helpful assistant focused on technical topics.<|im_end|>
<|im_start|>user
Can you explain what a chat template is?<|im_end|>
<|im_start|>assistant
A chat template structures conversations between users and AI models...<|im_end|>
<|im_start|>user
How do I use it ?<|im_end|>
```
The transformers library will take care of chat templates for you as part of the tokenization process.

#### Messages to prompt

The easiest way to ensure your LLM receives a conversation correctly formatted is to use the chat_template from the model’s tokenizer.

```
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
```

To convert the previous conversation into a prompt, we load the tokenizer and call apply_chat_template:

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

The rendered_prompt returned by this function is now ready to use as the input for the model you chose!

## What are Tools?

One crucial aspect of AI Agents is their ability to take actions. As we saw, this happens through the use of Tools.

By giving your Agent the right Tools—and clearly describing how those Tools work—you can dramatically increase what your AI can accomplish.A 

Tool is a function given to the LLM. This function should fulfill a clear objective. you can in fact create a tool for any use case! For instance, if you need to perform arithmetic, giving a calculator tool to your LLM will provide better results than relying on the native capabilities of the model.

For instance, if you need to perform arithmetic, giving a calculator tool to your LLM will provide better results than relying on the native capabilities of the model. For instance, if you ask an LLM directly (without a search tool) for today’s weather, the LLM will potentially hallucinate random weather.

A Tool contains:
* A textual description of what the function does.
* A Callable (something to perform an action).
* Arguments with typings.
* (Optional) Outputs with typings.

### How do tools work?

LLMs, as we saw, can only receive text inputs and generate text outputs. They have no way to call tools on their own. What we mean when we talk about providing tools to an Agent, is that we teach the LLM about the existence of tools, and ask the model to generate text that will invoke tools when it needs to. For example, if we provide a tool to check the weather at a location from the Internet, and then ask the LLM about the weather in Paris, the LLM will recognize that question as a relevant opportunity to use the “weather” tool we taught it about. The LLM will generate text, in the form of code, to invoke that tool. It is the responsibility of the Agent to parse the LLM’s output, recognize that a tool call is required, and invoke the tool on the LLM’s behalf. The output from the tool will then be sent back to the LLM, which will compose its final response for the user.

#### How do we give tools to an LLM?

Use the system prompt to provide textual descriptions of available tools to the model:

```
system_message = """You are a helpful assistant with access to the following tools: weather, calculator, and search. You can use these tools:
{tool_description}
"""
```

We will implement a simplified calculator tool that will just multiply two integers. This could be our Python implementation:

```
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

So our tool is called calculator, it multiplies two integers, and it requires the following inputs:

a (int): An integer.
b (int): An integer.
The output of the tool is another integer number that we can describe like this:

(int): The product of a and b.
All of these details are important. Let’s put them together in a text string that describes our tool for the LLM to understand.

```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

This textual description is what we want the LLM to know about the tool.

#### Auto-formatting Tool sections

Our tool was written in Python, and the implementation already provides everything we need:

* A descriptive name of what it does: calculator
* A longer description, provided by the function’s docstring comment: Multiply two integers.
* The inputs and their type: the function clearly expects two ints.
* The type of the output.

We could provide the Python source code as the specification of the tool for the LLM, but the way the tool is implemented does not matter. All that matters is its name, what it does, the inputs it expects and the output it provides.

We will leverage Python’s introspection features to leverage the source code and build a tool description automatically for us. All we need is that the tool implementation uses type hints, docstrings, and sensible function names. We will write some code to extract the relevant portions from the source code.

After we are done, we’ll only need to use a Python decorator to indicate that the calculator function is a tool:

```
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

With the tool decorator implementation, we will be able to retrieve the following text automatically from the source code:

```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

This is a simplified example, but it shows the power of tools. We can now use the calculator tool in our system prompt:
```
class Tool:
    """
    A class representing a reusable piece of code (Tool).
    
    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of argument.
        outputs (str or list): The return type(s) of the wrapped function.
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 func: callable, 
                 arguments: list,
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """
        Return a string representation of the tool, 
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        
        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)

def tool(func):
    """
    A decorator that creates a Tool instance from the given function.
    """
    # Get the function signature
    signature = inspect.signature(func)
    
    # Extract (param_name, param_annotation) pairs for inputs
    arguments = []
    for param in signature.parameters.values():
        annotation_name = (
            param.annotation.__name__ 
            if hasattr(param.annotation, '__name__') 
            else str(param.annotation)
        )
        arguments.append((param.name, annotation_name))
    
    # Determine the return annotation
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:
        outputs = "No return annotation"
    else:
        outputs = (
            return_annotation.__name__ 
            if hasattr(return_annotation, '__name__') 
            else str(return_annotation)
        )
    
    # Use the function's docstring as the description (default if None)
    description = func.__doc__ or "No description provided."
    
    # The function name becomes the Tool name
    name = func.__name__
    
    # Return a new Tool instance
    return Tool(
        name=name, 
        description=description, 
        func=func, 
        arguments=arguments, 
        outputs=outputs
    )
```

The decorator description is injected in the system prompt. Taking the example with which we started this section, here is how it would look like after replacing the tools_description:


## Understanding AI Agents through the Thought-Action-Observation Cycle

Agents work in a continuous cycle of: thinking (Thought) → acting (Act) and observing (Observe).

Let’s break down these actions together:

* Thought: The LLM part of the Agent decides what the next step should be.
* Action: The agent takes an action, by calling the tools with the associated arguments.
* Observation: The model reflects on the response from the tool.

### The Thought-Action-Observation Cycle

In many Agent frameworks, the rules and guidelines are embedded directly into the system prompt, ensuring that every cycle adheres to a defined logic.

```
system_message="""You are an AI assistant designed to help users efficiently and accurately. Your primary goal is to provide helpful, precise, and clear responses.

You have access to the following tools:
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int

You should think step by step in order to fulfill the objective with a reasoning devided in Thought/Action/Observation that can repeat multiple times if needed.

You should first reflect with ‘Thought: {your_thoughts}’ on the current situation, then (if necessary ), call a tool with the proper JSON formatting ‘Action: {JSON_BLOB}’,or you print your final answer starting with the prefix ‘Final Answer:’
"""
```

### The weather Agent

A user asks the agent: “What’s the weather like in New York today?”
Alfred’s job is to answer this query using a weather API tool.

Thought: “The user needs current weather information for New York. I have access to a tool that fetches weather data. First, I need to call the weather API to get up-to-date details.”
Action: Based on its reasoning and the fact that Alfred knows about a get_weather tool, Alfred prepares a JSON-formatted command that calls the weather API tool.
Observation: After the tool call, Alfred receives an observation. This might be the raw weather data from the API such as: “Current weather in New York: partly cloudy, 15°C, 60% humidity.” This observation is then added to the prompt as additional context. It functions as real-world feedback, confirming whether the action succeeded and providing the needed details.
Updated thought: “Now that I have the weather data for New York, I can compile an answer for the user.”
Final Action: 
Thought: I have the weather data now. The current weather in New York is partly cloudy with a temperature of 15°C and 60% humidity.”
Final answer : The current weather in New York is partly cloudy with a temperature of 15°C and 60% humidity.

### ReAct cycle

The interplay of Thought, Action, and Observation empowers AI agents to solve complex tasks iteratively.
By understanding and applying these principles, you can design agents that not only reason about their tasks but also effectively utilize external tools to complete them, all while continuously refining their output based on environmental feedback.

## Thought: Internal Reasoning and the Re-Act Approach

Type of Thought:
* Planning: “I need to break this task into three steps: 1) gather data, 2) analyze trends, 3) generate report”
* Analysis: “Based on the error message, the issue appears to be with the database connection parameters”
* Decision Making: “Given the user’s budget constraints, I should recommend the mid-tier option”
* Problem Solving: “To optimize this code, I should first profile it to identify bottlenecks”
* Memory Integration: “The user mentioned their preference for Python earlier, so I’ll provide examples in Python”
* Self-Reflection: “My last approach didn’t work well, I should try a different strategy”
* Goal Setting: “To complete this task, I need to first establish the acceptance criteria”
* Prioritization: “The security vulnerability should be addressed before adding new features”

### Action: External Tool Invocation

Type of Action:
* API Call: “I need to call the weather API to get the current weather in New York”
* Function Call: “I need to call the get_weather function with the appropriate arguments”

### The Re-Act Approach

A key method is the ReAct approach, which is the concatenation of “Reasoning” (Think) with “Acting” (Act).

ReAct is a simple prompting technique that appends “Let’s think step by step” before letting the LLM decode the next tokens.

Indeed, prompting the model to think “step by step” encourages the decoding process toward next tokens that generate a plan, rather than a final solution, since the model is encouraged to decompose the problem into sub-tasks.

This allows the model to consider sub-steps in more detail, which in general leads to less errors than trying to generate the final solution directly.

Models like Deepseek R1 or OpenAI's o1, which have been fine-tuned to "think before answering".
These models have been trained to always include specific thinking sections (enclosed between <think> and </think> special tokens). This is not just a prompting technique like ReAct, but a training method where the model learns to generate these sections after analyzing thousands of examples that show what we expect it to do.

## Actions: Enabling the Agent to Engage with Its Environment

Actions are the concrete steps an AI agent takes to interact with its environment.

Whether it’s browsing the web for information or controlling a physical device, each action is a deliberate operation executed by the agent.

Types of Agent Actions:
* JSON Agent: The Action to take is specified in JSON format.
* Code Agent: The Agent writes a code block that is interpreted externally.
* Function-calling Agent: It is a subcategory of the JSON Agent which has been fine-tuned to generate a new message for each action.

Type of Action:
* Information Gathering: Performing web searches, querying databases, or retrieving documents.
* Tool Usage: Making API calls, running calculations, and executing code.
* Environment Interaction: Manipulating digital interfaces or controlling physical devices.
* Communication: Engaging with users via chat or collaborating with other agents.

One crucial part of an agent is the ability to STOP generating new tokens when an action is complete, and that is true for all formats of Agent: JSON, code, or function-calling. This prevents unintended output and ensures that the agent’s response is clear and precise.

### The Stop and Parse Approach

One key method for implementing actions is the stop and parse approach. This method ensures that the agent’s output is structured and predictable:

Generation in a Structured Format: The agent outputs its intended action in a clear, predetermined format (JSON or code).
Halting Further Generation: Once the action is complete, the agent stops generating additional tokens. This prevents extra or erroneous output.
Parsing the Output: An external parser reads the formatted action, determines which Tool to call, and extracts the required parameters.

### Code Agents

The idea is: instead of outputting a simple JSON object, a Code Agent generates an executable code block—typically in a high-level language like Python.

For example, a Code Agent tasked with fetching the weather might generate the following Python snippet:

```
# Code Agent Example: Retrieve Weather Information
def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "No weather information available")
    else:
        return "Error: Unable to fetch weather data."

# Execute the function and prepare the final answer
result = get_weather("New York")
final_answer = f"The current weather in New York is: {result}"
print(final_answer)
```

## Observe: Integrating Feedback to Reflect and Adapt

Observations are how an Agent perceives the consequences of its actions.

They are signals from the environment—whether it’s data from an API, error messages, or system logs—that guide the next cycle of thought.

In the observation phase, the agent:

* Collects Feedback: Receives data or confirmation that its action was successful (or not).
* Appends Results: Integrates the new information into its existing context, effectively updating its memory.
* Adapts its Strategy: Uses this updated context to refine subsequent thoughts and actions.

Type of Observation:
* System Feedback: Error messages, success notifications, status codes
* Data Changes: Database updates, file system modifications, state changes
* Environmental Data: Sensor readings, system metrics, resource usage
* Response Analysis: API responses, query results, computation outputs
* Time-based Events: Deadlines reached, scheduled tasks completed

## Dummy Agent Library

{% jupyter_notebook "../../assets/2025-02/building_agents/dummy_agent_library.ipynb" %}

## Let’s Create Our First Agent Using smolagents

### What is smolagents?

Smolagents is a library that focuses on codeAgent, a kind of agent that performs “Actions” through code blocks, and then “Observes” results by executing the code.

Lightweight library is designed for simplicity, but it abstracts away much of the complexity of building an Agent, allowing you to focus on designing your agent’s behavior.

We provided our agent with an Image generation tool and asked it to generate an image of a cat.

The agent inside smolagents is going to have the same behaviors as the custom one we built previously: it’s going to think, act and observe in cycle until it reaches a final answer.

### Let’s build our Agent!

{% jupyter_notebook "../../assets/2025-02/building_agents/smol_agent.ipynb" %}

## Fine tuning LLM for function calling

Function-calling is a way for an LLM to take actions on its environment. It was first introduced in GPT-4, and was later reproduced in other models.

Just like the tools of an Agent, function-calling gives the model the capacity to take an action on its environment. However, the function calling capacity is learned by the model, and relies less on prompting than other agents techniques.

During Unit 1, the Agent didn’t learn to use the Tools, we just provided the list, and we relied on the fact that the model was able to generalize on defining a plan using these Tools.

[Huggingface course](https://huggingface.co/learn/nlp-course/chapter11/4?fw=pt)

{% jupyter_notebook "https://colab.research.google.com/#fileId=https%3A//huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb" %}

Start from `https://huggingface.co/google/gemma-2-2b-it`, a model that has been trained for instruction following (instruction tuned). Instruction tuned models introduce extra tokens to the vocabulary to help the model understand the instructions; examples are `<start_of_turn>` and `<end_of_turn>` in the case of gemma.

The tokenizer provides a built-in chat template that can be used to format the conversation as a list of messages. As an example, the following chart:

```
chat = [
    { "role": "user", "content": "Write a hello world program" },
]
```

Ran through the tokenizer's chat template:
```
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
```

Will output:
```
<bos><start_of_turn>user
Write a hello world program<end_of_turn>
<start_of_turn>model
```

### NousResearch/hermes-function-calling-v1 DataSet

A popular Dataset for function calling with 11578 examples. Here's an example:

```json
[
    {
        "from": "system",
        "value": "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. <tools> [{'type': 'function', 'function': {'name': 'get_camera_live_feed', 'description': 'Retrieves the live feed from a specified security camera.', 'parameters': {'type': 'object', 'properties': {'camera_id': {'type': 'string', 'description': 'The unique identifier for the camera.'}, 'stream_quality': {'type': 'string', 'description': 'The desired quality of the live stream.', 'enum': ['720p', '1080p', '4k']}}, 'required': ['camera_id']}}}, {'type': 'function', 'function': {'name': 'list_all_cameras', 'description': 'Lists all the security cameras connected to the home network.', 'parameters': {'type': 'object', 'properties': {'include_offline': {'type': 'boolean', 'description': 'Whether to include cameras that are currently offline.', 'default': False}}, 'required': []}}}, {'type': 'function', 'function': {'name': 'record_camera_feed', 'description': 'Starts recording the live feed from a specified security camera.', 'parameters': {'type': 'object', 'properties': {'camera_id': {'type': 'string', 'description': 'The unique identifier for the camera.'}, 'duration': {'type': 'integer', 'description': 'The duration in minutes for which to record the feed.', 'default': 60}}, 'required': ['camera_id']}}}, {'type': 'function', 'function': {'name': 'get_recorded_feed', 'description': 'Retrieves a previously recorded feed from a specified security camera.', 'parameters': {'type': 'object', 'properties': {'camera_id': {'type': 'string', 'description': 'The unique identifier for the camera.'}, 'start_time': {'type': 'string', 'description': 'The start time of the recording to retrieve, in ISO 8601 format.'}, 'end_time': {'type': 'string', 'description': 'The end time of the recording to retrieve, in ISO 8601 format.'}}, 'required': ['camera_id', 'start_time', 'end_time']}}}, {'type': 'function', 'function': {'name': 'pan_tilt_camera', 'description': 'Controls the pan and tilt functions of a PTZ (Pan-Tilt-Zoom) security camera.', 'parameters': {'type': 'object', 'properties': {'camera_id': {'type': 'string', 'description': 'The unique identifier for the PTZ camera.'}, 'pan_angle': {'type': 'integer', 'description': 'The angle in degrees to pan the camera. Positive values pan right, negative values pan left.'}, 'tilt_angle': {'type': 'integer', 'description': 'The angle in degrees to tilt the camera. Positive values tilt up, negative values tilt down.'}}, 'required': ['camera_id', 'pan_angle', 'tilt_angle']}}}] </tools> For each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema: <tool_call> {'arguments': <args-dict>, 'name': <function-name>} </tool_call> "
    },
    {
        "from": "human",
        "value": "I've recently installed a new security system at my home, and I want to ensure everything is functioning as it should. Specifically, I'd like to start by checking the live feed from the camera located at the front door to monitor any activity. The camera has a unique identifier, which I've already configured to be "front_door." I'd prefer to view the live stream in high definition, so a 1080p quality would be ideal. Could you please call the appropriate function to retrieve the live feed from my front door camera in 1080p quality and provide me with the link to the stream? Following this, I would also like to record the live feed from this camera for the next 30 minutes. This is to test the recording feature and to keep an archived copy for security purposes. Please initiate the recording function for the "front_door" camera with a recording duration of 30 minutes. Lastly, as part of my routine surveillance checks, I need to review footage from yesterday between 3 PM and 5 PM. The camera "front_garden" should have the recording for that period. I need to retrieve this specific recorded feed. The start time for the recording was at 15:00 on April 22, 2023, and it ended at 17:00 on the same day. To summarize, I request the execution of the following functions with the provided details: 1. Retrieve the live feed from the "front_door" camera at 1080p quality. 2. Start a 30-minute recording of the live feed from the "front_door" camera. 3. Retrieve the recorded feed from the "front_garden" camera, focusing on the time period between 15:00 and 17:00 on April 22, 2023. Thank you for assisting with the management of my home security camera feeds."
    },
    {
        "from": "gpt",
        "value": "<tool_call> {'arguments': {'camera_id': 'front_door', 'stream_quality': '1080p'}, 'name': 'get_camera_live_feed'} </tool_call> <tool_call> {'arguments': {'camera_id': 'front_door', 'duration': 30}, 'name': 'record_camera_feed'} </tool_call> <tool_call> {'arguments': {'camera_id': 'front_garden', 'start_time': '2023-04-22T15:00:00Z', 'end_time': '2023-04-22T17:00:00Z'}, 'name': 'get_recorded_feed'} </tool_call> "
    }
]
```

Now, this dataset does not contain the thinking part, so we need to add it. See next.


### hermes-function-calling-thinking-V1 DataSet

Contains a single split called "train" with 3570 examples:

```json
[ 
    { 
        "content": "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'get_stock_price', 'description': 'Get the current stock price of a company', 'parameters': {'type': 'object', 'properties': {'company': {'type': 'string', 'description': 'The name of the company'}}, 'required': ['company']}}}, {'type': 'function', 'function': {'name': 'get_movie_details', 'description': 'Get details about a movie', 'parameters': {'type': 'object', 'properties': {'title': {'type': 'string', 'description': 'The title of the movie'}}, 'required': ['title']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{tool_call}\n</tool_call>", 
        "role": "system" 
    }, 
    { "content": "Hi, can you tell me the current stock price of Apple?", 
        "role": "human" 
    }, 
    { 
        "content": "<think>Alright, the user asked for the current stock price of Apple. Looking at the tools available, there's a function called get_stock_price which exactly fits this request. I'll need to call this function with the parameter company set to 'Apple' to retrieve the information the user is asking for. This seems like the most straightforward and appropriate action to take.\n</think><tool_call>\n{'name': 'get_stock_price', 'arguments': {'company': 'Apple'}}\n</tool_call>", 
        "role": "model" 
    }, 
    { 
        "content": "<tool_response>\n{'stock_price': '$150.75'}\n</tool_response>", 
        "role": "tool" 
    }, 
    { "content": "The current stock price of Apple is $150.75.", 
        "role": "model" 
    }, 
    { 
        "content": "That's great. Can you also tell me the stock price for Microsoft?", 
        "role": "human" 
    }, 
    { 
        "content": "<think>Alright, let me try to figure out how to handle this user's request. The user first asked about Apple's stock price, and I responded by calling the get_stock_price function with 'Apple' as the argument. The tool provided the price, and I passed that information back to the user.\n\nNow, the user is asking for the stock price of Microsoft. I need to determine the appropriate action here. Looking at the tools available, there's still the get_stock_price function, which is designed to fetch current stock prices. Since the user is requesting another company's stock price, it makes sense to use the same function again but with the company name changed to Microsoft. \n\nI shouldn't assume any other tools are needed because the available functions are specifically tailored for stock prices and movie details. In this case, the relevant function is get_stock_price. I should ensure that the function is called with the correct parameter, 'Microsoft' in this case, to retrieve the accurate information the user is seeking. This approach is efficient and aligns directly with the user's request without making unnecessary assumptions or using unrelated tools.\n</think><tool_call>\n{'name': 'get_stock_price', 'arguments': {'company': 'Microsoft'}}\n</tool_call>", 
        "role": "model" 
    }, 
    { 
        "content": "<tool_response>\n{'stock_price': '$210.22'}\n</tool_response>", 
        "role": "tool" 
    }, 
    { 
        "content": "The current stock price of Microsoft is $210.22.", 
        "role": "model" 
    }, 
    { 
        "content": "Thank you for the information.", 
        "role": "human" 
    }, 
    { 
        "content": "You're welcome! If you have any other questions, feel free to ask.", 
        "role": "model" 
    } 
]

```
