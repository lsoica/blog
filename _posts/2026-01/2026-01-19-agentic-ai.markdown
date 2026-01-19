---
title:  "Agentic AI"
description: >-
  Exploring the capabilities of agentic artificial intelligence.
author: lso
date:   2026-01-19 11:08:03 +0200
categories: [Blogging, Tutorial]
tags: [agentic-ai, aiagents, langchain, langgraph, crewai, autogen, beeai]
pin: false
media_subpath: '/posts/20260119'
---
# Agentic AI

Agentic AI Systems: These are autonomous systems that can make decisions and take actions to achieve specific goals. Multi-agent systems utilize specialized agents working in parallel to enhance scalability, modularity, and fault tolerance.

## Frameworks

### CrewAI

CrewAI is a **cutting-edge framework** that empowers us to create and manage teams of **autonomous AI agents** designed to collaborate on complex tasks. Think of it as our ultimate toolkit for assembling a team of virtual experts, where each member plays a **specific role**, uses **unique tools**, and works toward **clear goals**. These agents aren’t just working in isolation; they collaborate, communicate, and solve problems as a synchronized team, enabling us to achieve more than ever before.   

Crews: groups of role-playing agents that interact and collaborate to achieve shared objectives. Each agent is:

* Assigned a Role: Just like in a real team, every agent has a specialized function, whether it’s planning, executing, or coordinating tasks.
* Equipped with Tools: Agents are provided with the resources they need to perform their roles effectively.
* Directed by Goals: Clear objectives ensure that every agent’s efforts align with the crew’s mission.

### LangGraph

Allows for structured workflows using directed graphs, providing fine-grained control over agent interactions.

Complex Workflow Automation: It excels in automating multi-step processes that require advanced memory management and structured workflows. This makes it ideal for industries like finance or healthcare, where precise control over interactions is crucial.

Fine-Grained Control: LangGraph allows for detailed control over agent interactions using directed graphs. Each node represents a step in the workflow, and edges define the flow of information, making it perfect for applications that need specific decision-making paths.

Error Recovery and State Management: It provides robust error recovery mechanisms and state management capabilities, which are essential for applications that require high reliability and the ability to handle failures gracefully.

Integration with Tools and LLMs: LangGraph integrates well with various tools and large language models (LLMs), allowing for complex interactions and the ability to leverage external resources effectively.

Custom Agent Workflows: If you need to build custom workflows tailored to specific requirements, LangGraph's flexibility in defining nodes and edges enables the creation of unique agent interactions that can adapt to changing conditions.

### AutoGen

Designed for conversational interactions, enabling real-time collaboration between agents and humans.

Conversational Interactions: It excels in environments that require real-time, dialogue-driven interactions between agents and between agents and humans. This makes it ideal for applications like virtual assistants or customer service bots.

Human-in-the-Loop Workflows: AutoGen is particularly effective in situations where human oversight is necessary. It allows users to step in to guide or approve actions, making it suitable for tasks that blend automation with human review, such as content moderation or technical support.

Rapid Prototyping: The framework is designed for quick prototyping, enabling developers to create and test conversational agents efficiently. This is beneficial in educational settings or for developing proof-of-concept applications.

Role-Based Collaboration: AutoGen allows for the definition of agent personas, enabling agents to take on specific roles and responsibilities within a conversation. This helps in simulating team dynamics and collaborative problem-solving.

Dynamic Task Delegation: It supports dynamic task delegation, where agents can clarify goals, ask questions, and delegate work in real-time, making it suitable for scenarios that require adaptability and responsiveness.

### BeeAI

Built for enterprise-grade deployment, offering robust features for scalability and operational management.

Enterprise-Grade Deployment: It is designed for robust, scalable applications that require high reliability and operational management. This makes it ideal for large organizations looking to implement AI solutions that can handle significant workloads.

Complex Multi-Agent Coordination: BeeAI excels in environments where multiple agents need to work together seamlessly. Its architecture supports complex interactions and coordination among agents, making it suitable for applications that require intricate workflows.

Tool Integration: The framework integrates well with various tools and services, allowing for enhanced functionality and the ability to leverage existing resources. This is particularly useful in enterprise settings where different systems need to communicate effectively.

Memory Management and State Persistence: BeeAI offers advanced memory management capabilities, enabling agents to retain context and state across interactions. This is crucial for applications that require continuity and context awareness, such as customer support systems.

Scalability and Fault Tolerance: It is built to handle failures gracefully, ensuring that the system remains operational even in the face of errors. This makes BeeAI suitable for mission-critical applications where downtime is not an option.

## Design patterns

### Sequential Flows

This design pattern involves structuring workflows in a linear sequence, where each step depends on the completion of the previous one. It is useful for straightforward processes that do not require complex decision-making.

```python
class ChainState(TypedDict):
    job_description: str
    resume_summary: str
    cover_letter: str

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_resume_summary(state: ChainState) -> ChainState:
    prompt = f"""
You're a resume assistant. Read the following job description and summarize the key qualifications and experience the ideal candidate should have, phrased as if from the perspective of a strong applicant's resume summary.

Job Description:
{state['job_description']}
"""

    response = llm.invoke(prompt)

    return {**state, "resume_summary": response.content}

def generate_cover_letter(state: ChainState) -> ChainState:
    prompt = f"""
You're a cover letter writing assistant. Using the resume summary below, write a professional and personalized cover letter for the following job.

Resume Summary:
{state['resume_summary']}

Job Description:
{state['job_description']}
"""

    response = llm.invoke(prompt)

    return {**state, "cover_letter": response.content}

workflow = StateGraph(ChainState)
workflow.add_node("generate_resume_summary", generate_resume_summary)
workflow.add_node("generate_cover_letter", generate_cover_letter)

workflow.set_entry_point("generate_resume_summary")
workflow.add_edge("generate_resume_summary", "generate_cover_letter")
workflow.set_finish_point("generate_cover_letter")

app = workflow.compile()

input_state = {
        "job_description": "We are looking for a data scientist with experience in machine learning, NLP, and Python. Prior work with large datasets and experience deploying models into production is required."
}

result = app.invoke(input_state)
result['resume_summary']
```

### Parallelization

This design pattern focuses on executing multiple tasks simultaneously, enhancing efficiency and reducing processing time. It is particularly beneficial in scenarios where tasks can be performed independently.

```python
class State(TypedDict):
    text: str
    french: str
    spanish: str
    japanese: str
    combined_output: str

def translate_french(state: State) -> dict:
    response = llm.invoke(f"Translate the following text to French:\n\n{state['text']}")
    return {"french": response.content.strip()}

def translate_spanish(state: State) -> dict:
    response = llm.invoke(f"Translate the following text to Spanish:\n\n{state['text']}")
    return {"spanish": response.content.strip()}

def translate_japanese(state: State) -> dict:
    response = llm.invoke(f"Translate the following text to Japanese:\n\n{state['text']}")
    return {"japanese": response.content.strip()}

def aggregator(state: State) -> dict:
    combined = f"Original Text: {state['text']}\n\n"
    combined += f"French: {state['french']}\n\n"
    combined += f"Spanish: {state['spanish']}\n\n"
    combined += f"Japanese: {state['japanese']}\n"
    return {"combined_output": combined}

graph = StateGraph(State)

graph.add_node("translate_french", translate_french)
graph.add_node("translate_spanish", translate_spanish)
graph.add_node("translate_japanese", translate_japanese)
graph.add_node("aggregator", aggregator)

# Connect parallel nodes from START
graph.add_edge(START, "translate_french")
graph.add_edge(START, "translate_spanish")
graph.add_edge(START, "translate_japanese")

# Connect all translation nodes to the aggregator
graph.add_edge("translate_french", "aggregator")
graph.add_edge("translate_spanish", "aggregator")
graph.add_edge("translate_japanese", "aggregator")

graph.add_edge("aggregator", END)

app = graph.compile()

input_text = {
        "text": "Good morning! I hope you have a wonderful day."
}

result = app.invoke(input_text)
```

### Routing

This pattern allows for conditional branching in workflows, enabling the system to make decisions based on specific criteria. It helps in directing the flow of tasks and information based on the current state or inputs.

```python
class RouterState(TypedDict):
    user_input: str
    task_type: str
    output: str

class Router(BaseModel):
    role: str = Field(..., description="Decide whether the user wants to summarize a passage  ouput 'summarize'  or translate text into French oupput translate.")

llm_router=llm.bind_tools([Router])

def router_node(state: RouterState) -> RouterState:
    routing_prompt = f"""
    You are an AI task classifier.
    
    Decide whether the user wants to:
    - "summarize" a passage
    - or "translate" text into French
    
    Respond with just one word: 'summarize' or 'translate'.
    
    User Input: "{state['user_input']}"
    """

    response = llm_router.invoke(routing_prompt)

    return {**state, "task_type": response.tool_calls[0]['args']['role']} # This becomes the next node's name!

def router(state: RouterState) -> str:
    return state['task_type']

def summarize_node(state: RouterState) -> RouterState:
    prompt = f"Please summarize the following passage:\n\n{state['user_input']}"
    response = llm.invoke(prompt)
    
    return {**state, "task_type": "summarize", "output": response.content}

def translate_node(state: RouterState) -> RouterState:
    prompt = f"Translate the following text to French:\n\n{state['user_input']}"
    response = llm.invoke(prompt)

    return {**state, "task_type": "translate", "output": response.content}

workflow = StateGraph(RouterState)

workflow.add_node("router", router_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("translate", translate_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges("router", router, {
    "summarize": "summarize",
    "translate": "translate"
})

workflow.set_finish_point("summarize")
workflow.set_finish_point("translate")

app = workflow.compile()

input_text = {
        "user_input": "Can you translate this sentence: I love programming?"
    }

result = app.invoke(input_text)

print(result[ 'output'])
print(result['task_type'])

input_text = {
        "user_input": "Can you summarize this sentence: I love programming so much it is the best thing ever. All I want to do is programming?"
    }

result = app.invoke(input_text)

print(result[ 'output'])
print(result['task_type'])
```

### Evaluator-Optimizer Pattern

This pattern is used to refine AI models through feedback loops. It involves a generator that produces outputs, which are then evaluated. If the output does not meet the desired criteria, feedback is provided to the generator for improvement. This iterative process continues until the output is satisfactory.

Also called Reflection or Self-Refinement pattern, it is particularly useful in scenarios where continuous improvement of model outputs is required, such as content generation or decision-making tasks.

```python
grades = Literal[
    "ultra-conservative", 
    "conservative", 
    "moderate", 
    "aggressive", 
    "high risk"
]

class State(TypedDict):
    investment_plan: str
    investor_profile: str
    target_grade: grades
    feedback: str
    grade: grades
    n: int = 0

grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an investment advisor. Given the investor’s profile and their proposed plan,"
     "choose exactly one risk classification from: ultra-conservative, conservative, moderate, aggressive, high risk."
     "Return ONLY the grade."
    ),
    ("user",
     "Investor profile:\n\n{investor_profile}\n\n"
    )
])

grade_pipe = grade_prompt | llm

def determine_target_grade(state: State):
    """Ask the LLM to pick the best-fitting target_grade."""
    response = grade_pipe.invoke({
        "investor_profile": state["investor_profile"]
    })
    
    # return as a plain dict so LangGraph can merge it into the state
    return {"target_grade": response.content.lower()}

# inital generation, no feedback, only based on profile
cathie_wood_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are a bold, innovation-driven investment advisor inspired by Cathie Wood.

Your goal is to generate a high-conviction, forward-looking investment plan that embraces disruptive technologies,
emerging markets, and long-term growth potential. You are not afraid of short-term volatility as long as the upside is transformational.

Create an investment strategy tailored to the investor profile below. Prioritize innovation and high-reward opportunities,
such as artificial intelligence, biotechnology, blockchain, or renewable energy.

Respond with a concise investment plan in paragraph form.
"""
    ),
    ("human", "Investor profile:\n\n{investor_profile}")
])

cathie_wood_pipe = cathie_wood_prompt | llm

# evaluator output schema
class Feedback(BaseModel):
    grade: grades = Field(
        description="Classify the investment based on risk level, ranging from ultra-conservative to high risk."
    )
    feedback: str = Field(
        description="Provide reasoning for the risk classification assigned to the investment suggestion."
    )

ray_dalio_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """You are an investment advisor inspired by Ray Dalio's principles but with adaptive strategy generation.
Your goal is to create varied, scenario-aware investment plans that respond dynamically to economic conditions,
feedback, and the investor's evolving needs. You adapt your recommendations based on previous evaluations.

CORE PRINCIPLES:
- Environmental diversification across economic regimes (growth/inflation combinations)
- Risk parity weighting by volatility, not just dollar amounts
- Inflation-aware asset selection with real return focus
- Macroeconomic scenario planning and regime identification

ADAPTATION RULES based on feedback:
- If deemed "too conservative" → Increase growth equity allocation, add emerging markets, consider alternatives
- If deemed "too aggressive" → Add defensive assets, increase bond allocation, focus on dividend stocks
- If "lacks inflation protection" → Emphasize TIPS, commodities, REITs, international exposure
- If "too complex" → Simplify to core ETF strategy with clear rationale
- If "insufficient diversification" → Add geographic, sector, or alternative asset exposure

ECONOMIC SCENARIO ADJUSTMENTS:
- Rising inflation environment → Emphasize commodities, TIPS, real estate, reduce duration
- Stagflation concerns → Focus on energy, materials, international markets, inflation hedges
- Deflationary risks → Increase government bonds, high-quality corporate bonds, cash positions
- Growth acceleration → Favor technology, consumer discretionary, small-cap growth
- Economic uncertainty → Balance with "All Weather" approach using multiple asset classes

TARGETING 15% RETURNS through:
- Strategic overweighting of growth assets during favorable conditions
- Tactical allocation adjustments based on economic regime
- Alternative investments (REITs, commodities, international) for diversification
- Leverage consideration for qualified investors
- Regular rebalancing to capture volatility

Respond with a clear, actionable investment plan that reflects current economic conditions 
and adapts to the specific feedback provided. Vary your approach significantly based on 
the grade and feedback received.
"""
    ),
    ("human",
     """Investor profile:
{investor_profile}

Previous strategy grade: {grade}

Evaluator feedback: {feedback}

Based on this feedback, create a NEW investment strategy that addresses the concerns raised 
while targeting 15% returns. Make significant adjustments from any previous approach.
""")
])

ray_dalio_pipe = ray_dalio_prompt | llm

def investment_plan_generator(state: State) -> dict:
    """Prompts an LLM to generate or improve an investment plan based on the current state."""

    if state.get("feedback"):
        # use Ray Dalio–style generator when feedback is available
        response = ray_dalio_pipe.invoke({
            "investor_profile": state["investor_profile"],
            "grade": state["grade"],
            "feedback": state["feedback"]
        })
    else:
        # use Cathie Wood–style generator for initial plan
        response = cathie_wood_pipe.invoke({
            "investor_profile": state["investor_profile"]
        })

    return {"investment_plan": response.content}

# Warren Buffet style evaluation prompt
evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """You are an investment risk evaluator inspired by Warren Buffett's value investing philosophy.

Your task is to assess whether a proposed investment strategy aligns with conservative, value-driven principles 
that emphasize capital preservation, long-term stability, and sound business fundamentals. You should be 
skeptical of speculative investments, high-volatility assets, and short-term market trends.

RISK CLASSIFICATION LEVELS:
- ultra-conservative: Extremely safe, minimal risk of loss
- conservative: Low risk, prioritizes capital preservation  
- moderate: Balanced approach with acceptable risk-reward ratio
- aggressive: Higher risk for potentially greater returns
- high risk: Speculative investments with significant loss potential

EVALUATION CRITERIA:
- Business clarity: Is the investment easily understandable with transparent cash flows?
- Margin of safety: Does the investment price provide protection against downside risk?
- Capital preservation: Will this strategy protect wealth over the long term?
- Investor alignment: Does this match a conservative investor's risk tolerance and goals?
- Quality fundamentals: Are the underlying assets financially sound with competitive advantages?

Return your assessment in the following  format:
{{
  "grade": "<investment risk level>",
  "feedback": "<concise explanation of the assigned risk level and key reasoning>"
}}
"""
    ),
    ("human", 
     "Evaluate this investment plan:\n\n{investment_plan}\n\nFor this investor profile:\n\n{investor_profile}\n\nAnd provide feedback that matches this target risk level: {target_grade}")
])

# create the pipe with the structured output that outputs a Feedback object
buffett_evaluator_pipe = evaluator_prompt | llm.with_structured_output(Feedback)

def evaluate_plan(state: State):
    """LLM evaluates the investment plan"""

    # add one to the current count
    current_count = state.get('n', 0) + 1

    # get the evaluation result from the evaluator pipe
    evaluation_result = buffett_evaluator_pipe.invoke({
        "investment_plan": state["investment_plan"],
        "investor_profile": state["investor_profile"],
        "target_grade": state["target_grade"]
    })

    # return the grade and feedback in a dict
    return {"grade": evaluation_result.grade, "feedback": evaluation_result.feedback, "n": current_count}

def route_investment(state: State, iteration_limit: int = 5):
    """Route investment based on risk grade evaluation"""
    # get grades
    current_grade = state.get("grade", "MISSING")
    target_grade = state.get("target_grade", "MISSING")
    # check if grades match
    match = current_grade == target_grade

    # print out the tracked values
    print(f"=== ROUTING  ===")
    print(f"Current grade: '{current_grade}'")
    print(f"Target risk profile: '{target_grade}'")
    print(f"Match: {match}")
    print(f"Number of trials: {state['n']}")

    # routing logic
    if match: # grades match
        print("→ Routing to: Accepted")
        return "Accepted"
    elif state['n'] > iteration_limit: # review iterations exceeds limit
        print("→ Too many iterations, stopping")
        return "Accepted"
    else: # grades don't match
        print("→ Routing to: Rejected + Feedback")
        return "Rejected + Feedback"

# initialize StateGraph with the given State schema
optimizer_builder = StateGraph(State)

# add the setup, generator, and evaluator nodes
optimizer_builder.add_node("determine_target_grade", determine_target_grade)
optimizer_builder.add_node("investment_plan_generator", investment_plan_generator)
optimizer_builder.add_node("evaluate_plan", evaluate_plan)

# define the flow with edges
optimizer_builder.add_edge(START, "determine_target_grade")
optimizer_builder.add_edge("determine_target_grade", "investment_plan_generator")
optimizer_builder.add_edge("investment_plan_generator", "evaluate_plan")

# add conditional edge for reflection
optimizer_builder.add_conditional_edges(
    "evaluate_plan",
    lambda state: route_investment(state),
    {
        "Accepted": END,
        "Rejected + Feedback": "investment_plan_generator",
    },
)

# compile the workflow
optimizer_workflow = optimizer_builder.compile()

# invoke the workflow with an example investor profile
state = optimizer_workflow.invoke({
    "investor_profile": (
        "Age: 29\n"
        "Salary: $110,000\n"
        "Assets: $40,000\n"
        "Goal: Achieve financial independence by age 45\n"
        "Risk tolerance: High"
    )
})
```

### Orchestrator-Worker Design Pattern

This pattern focuses on coordinating multiple agents and tasks within a system. It allows for the management of workflows by defining how agents interact, the sequence of tasks, and the overall orchestration of processes. This is particularly useful in multi-agent systems where collaboration is key.

```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Dish schema for a single dish
class Dish(BaseModel):
    name: str = Field(
        description="Name of the dish (for example, Spaghetti Bolognese, Chicken Curry)."
    )
    ingredients: List[str] = Field(
        description="List of ingredients needed for this dish, separated by commas."
    )
    location: str = Field(
        description="The cuisine or cultural origin of the dish (for example, Italian, Indian, Mexican)."
    )

# Dishes schema for a list of Dish objects
class Dishes(BaseModel):
    sections: List[Dish] = Field(
        description="A list of grocery sections, one for each dish, with ingredients."
    )

# construct a prompt template
dish_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant that generates a structured grocery list.\n\n"
        "The user wants to prepare the following meals: {meals}\n\n"
        "For each meal, return a section with:\n"
        "- the name of the dish\n"
        "- a comma-separated list of ingredients needed for that dish.\n"
        "- the cuisine or cultural origin of the food"
    )
])

# use LCEL to pipe the prompt to an LLM with a structured output of Dishes
planner_pipe = dish_prompt | llm.with_structured_output(Dishes)

class State(TypedDict):
    meals: str  # The user's input listing the meals to prepare
    sections: List[Dish] # One section per meal/dish with ingredients
    completed_menu: Annotated[List[str], operator.add]  # Worker written dish guide chunks
    final_meal_guide: str  # Fully compiled, readable menu

def orchestrator(state: State):
    """Orchestrator that generates a structured dish list from the given meals."""

    # use the planner_pipe LLM to break the user's meal list into structured dish sections
    dish_descriptions = planner_pipe.invoke({"meals": state["meals"]})

    # return the list of dish sections to be passed to worker nodes
    return {"sections": dish_descriptions.sections}

chef_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a world-class chef from {location}.\n\n"
        "Please introduce yourself briefly and present a detailed walkthrough for preparing the dish: {name}.\n"
        "Your response should include:\n"
        "- Start with hello with your  name and culinary background\n"
        "- A clear list of preparation steps\n"
        "- A full explanation of the cooking process\n\n"
        "Use the following ingredients: {ingredients}."
    )
])

chef_pipe = chef_prompt | llm

class WorkerState(TypedDict):
    section: Dish
    completed_menu: Annotated[list, operator.add] # list with addition operators between elements

def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [Send("chef_worker", {"section": s}) for s in state["sections"]]

def chef_worker(state: WorkerState):
    """Worker node that generates the cooking instructions for one meal section."""

    # Use the language model to generate a meal preparation plan
    # The model receives the dish name, location, and ingredients from the current section
    meal_plan = chef_pipe.invoke({
        "name": state["section"].name,
        "location": state["section"].location,
        "ingredients": state["section"].ingredients
    })

    # Return the generated meal plan wrapped in a list under completed_sections
    # This will be merged into the main state using operator.add in LangGraph
    return {"completed_menu": [meal_plan.content]}

def synthesizer(state: State):
    """Synthesize full report from sections"""

    # list of completed sections
    completed_sections = state["completed_menu"]

    # format completed section to str to use as context for final sections
    completed_menu = "\n\n---\n\n".join(completed_sections)

    return {"final_meal_guide": completed_menu}

# instantiate the builder
orchestrator_worker_builder = StateGraph(State)

# add the nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("chef_worker", chef_worker)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["chef_worker"] # source node, routing function, list of allowed targets
)

# add the edges, connections between nodes
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_edge("chef_worker", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# compile the builder to get a complete workflow executable
orchestrator_worker = orchestrator_worker_builder.compile()

# invoke the workflow with a string of meals in a dict
state = orchestrator_worker.invoke({"meals": "Steak and eggs, tacos, and chili"})

# print the first 2000 characters of our final_meal_guide
pprint(state["final_meal_guide"][:2000])
```

## CrewAI

### Agent

Example:

```python
from crewai import Agent

research_agent = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge information and insights on any subject with comprehensive analysis',
  backstory="""You are an expert researcher with extensive experience in gathering, analyzing, and synthesizing information across multiple domains. 
  Your analytical skills allow you to quickly identify key trends, separate fact from opinion, and produce insightful reports on any topic. 
  You excel at finding reliable sources and extracting valuable information efficiently.""",
  verbose=True,
  allow_delegation=False,
  llm = llm,
  tools=[SerperDevTool()]
)
```

### Task

Tasks are like to-do items for our AI agents. Each task has specific instructions, details, and tools for the agent to follow and complete the job.

For example:

A task could ask an agent to "research the latest AI trends."
Another task could ask a different agent to "write a detailed report based on the research."

There are two ways tasks can run:

Sequential: Tasks are executed one after the other, like following a recipe step-by-step. Each task waits for the previous one to finish.
Hierarchical: Tasks are assigned based on agent skills or roles, and multiple tasks can run in parallel if they don’t depend on each other.

Example task definition:

```python
from crewai import Task

research_task = Task(
  description="Analyze the major {topic}, identifying key trends and technologies. Provide a detailed report on their potential impact.",
  agent=research_agent,
  expected_output="A detailed report on {topic}, including trends, emerging technologies, and their impact."
)
```

### Crew

The Crew object, which is the central orchestration mechanism in CrewAI. This crew brings together our specialized agents and their assigned tasks into a cohesive workflow.

The Crew constructor takes several important parameters:

agents: A list of the AI agents that will be part of this crew research_agent abd writer_agent
tasks: A list of specific tasks these agents will perform research_task and writer_task
process: Defines how tasks will be executed - in this case `Process.sequential means tasks will run one after another in the specified order (research first, then writing)
verbose: When set to True, this enables detailed logging, making it easier to follow the crew's execution and troubleshoot any issues
Once configured, you can start the entire workflow with a single command: crew.kickoff(), which will execute the tasks in sequence and return the final results.

Example crew definition and execution:

```python
from crewai import Crew, Process

crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, writer_task],
    process=Process.sequential,
    verbose=True 
)
result = crew.kickoff(inputs={"topic": "Latest Generative AI breakthroughs"})
```

### Structured Output

How outputs are structured plays a critical role in AI applications, especially those with multiple agents or complex data. Free-form text from language models can be difficult to parse, prone to ambiguity, and risky for downstream tasks. On the other hand, structured outputs (like JSON or objects with defined fields) ensure consistency and make data easier to extract and use.

CrewAI helps developers enforce structured outputs by letting them define schemas for task responses, which leads to more reliable and predictable outcomes. In multi-agent workflows, this structure ensures that one agent's output can be cleanly interpreted by the next, thus reducing miscommunication, data loss, and hallucinations. The result is smoother agent collaboration and easier system integration.

CrewAI leverages the Pydantic library to define these schemas, allowing developers to specify exactly what fields and data types are expected in task outputs. This not only improves the reliability of the AI system but also enhances maintainability and scalability as the application grows in complexity. Some benefits:

* Data validation: Ensures outputs conform to expected formats, reducing errors.
* Automatic type conversion
* Nested models and complex types
* Easy serialization to dict and json

#### Implementing structured output in a CrewAI task

```python
from typing import List
from pydantic import BaseModel
class Ingredient(BaseModel):
    name: str
    quantity: str
class MealPlan(BaseModel):
    meal_name: str
    ingredients: List[Ingredient]

blog_task = Task(
    description="Generate a catchy blog title and a short content about a topic.",
    expected_output="A JSON object with 'title' and 'content' fields.",
    agent=blog_agent,
    output_pydantic=BlogSummary
)
```

### Extending CrewAI with Custom Tools

CrewAI tools can be feed into both agents and tasks, allowing for flexible and powerful interactions. 

```python

# agent centric

agent_centric_agent = Agent(
    role="The Daily Dish Inquiry Specialist",
    goal="""Accurately answer customer questions about The Daily Dish restaurant. 
    You must decide whether to use the restaurant's FAQ PDF or a web search to find the best answer.""",
    backstory="""You are an AI assistant for 'The Daily Dish'.
    You have access to two tools: one for searching the restaurant's FAQ document and another for searching the web.
    Your job is to analyze the user's question and choose the most appropriate tool to find the information needed to provide a helpful response.""",
    tools=[pdf_search_tool, web_search_tool],
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# or task centric

faq_search_task = Task(
    description="Search the restaurant's FAQ PDF for information related to the customer's query: '{customer_query}'.",
    expected_output="A snippet of the most relevant information from the PDF, or a statement that the information was not found.",
    tools=[pdf_search_tool], # Tool assigned directly to the task
    agent=task_centric_agent
)
```

## BeeAI

## AutoGen
