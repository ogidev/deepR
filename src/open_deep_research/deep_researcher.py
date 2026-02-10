"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
from typing import Any, Dict, List, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    geneticist_system_prompt,
    lead_researcher_prompt,
    negotiation_synthesis_prompt,
    predictive_cognition_system_prompt,
    research_system_prompt,
    specialist_convergence_instructions,
    specialist_critique_round_instructions,
    specialist_proposal_round_instructions,
    systems_theorist_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    Hypothesis,
    HypothesesBundle,
    NegotiationState,
    QuerySpecialist,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorSpecialistQuery,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def handle_specialist_query(
    specialist_role: str,
    question: str,
    state: SupervisorState,
    config: RunnableConfig
) -> str:
    """Handle a direct query from supervisor to a specific specialist.
    
    Args:
        specialist_role: The role of the specialist to query (geneticist, systems_theorist, predictive_cognition)
        question: The specific question to ask the specialist
        state: Current supervisor state with research context
        config: Runtime configuration with model settings
        
    Returns:
        String response from the specialist
    """
    # Step 1: Select appropriate specialist prompt
    if specialist_role == "geneticist":
        system_prompt_template = geneticist_system_prompt
    elif specialist_role == "systems_theorist":
        system_prompt_template = systems_theorist_system_prompt
    elif specialist_role == "predictive_cognition":
        system_prompt_template = predictive_cognition_system_prompt
    else:
        return f"Error: Unknown specialist role '{specialist_role}'"
    
    # Step 2: Prepare context with research brief and notes
    notes = state.get("notes", [])
    raw_notes = state.get("raw_notes", [])
    notes_context = "\n".join(notes + raw_notes)[:10000]
    
    # Step 3: Format prompt for direct consultation
    system_prompt = system_prompt_template.format(
        research_brief=state.get("research_brief", ""),
        notes=notes_context,
        current_round=1,
        max_rounds=1,
        round_purpose="Direct consultation from Research Supervisor",
        additional_instructions=f"""
<Supervisor Direct Query>
The Research Supervisor has a specific question for you outside the normal negotiation process.
Provide a focused expert response from your specialist perspective.

Question: {question}
</Supervisor Direct Query>
"""
    )
    
    # Step 4: Configure and invoke specialist model
    configurable = Configuration.from_runnable_config(config)
    model_config = {
        "model": configurable.negotiation_model or configurable.research_model,
        "max_tokens": configurable.negotiation_max_tokens,
        "api_key": get_api_key_for_model(
            configurable.negotiation_model or configurable.research_model, 
            config
        ),
        "tags": ["langsmith:nostream"]
    }
    
    model = configurable_model.with_config(model_config)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    response = await model.ainvoke(messages)
    return str(response.content)


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, specialist queries, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, QuerySpecialist, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle QuerySpecialist calls (direct specialist queries)
    query_specialist_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "QuerySpecialist"
    ]
    
    if query_specialist_calls:
        for tool_call in query_specialist_calls:
            specialist_role = tool_call["args"]["specialist"]
            question = tool_call["args"]["question"]
            
            # Query the specialist directly
            specialist_response = await handle_specialist_query(
                specialist_role=specialist_role,
                question=question,
                state=state,
                config=config
            )
            
            # Record the interaction
            all_tool_messages.append(ToolMessage(
                content=f"[{specialist_role.replace('_', ' ').title()}] {specialist_response}",
                name="QuerySpecialist",
                tool_call_id=tool_call["id"]
            ))
            
            # Track in state for record-keeping
            if "specialist_queries" not in update_payload:
                update_payload["specialist_queries"] = []
                update_payload["specialist_responses"] = []
            
            update_payload["specialist_queries"].append({
                "specialist": specialist_role,
                "question": question,
                "iteration": research_iterations
            })
            update_payload["specialist_responses"].append({
                "specialist": specialist_role,
                "response": specialist_response,
                "iteration": research_iterations
            })
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()


###################
# Scientific Negotiation Subgraph
###################

# Constants for hypothesis parsing and synthesis
MAX_HYPOTHESIS_STATEMENT_LENGTH = 500  # Max chars for hypothesis statement
MAX_HYPOTHESIS_RATIONALE_LENGTH = 300  # Max chars for rationale
MAX_HYPOTHESIS_ASSUMPTION_LENGTH = 200  # Max chars for each assumption
MAX_HYPOTHESIS_VARIABLE_LENGTH = 100   # Max chars for each variable
MAX_FALLBACK_HYPOTHESES = 8  # Max hypotheses to include in fallback bundle
NUM_SPECIALISTS = 3  # Number of specialist agents in negotiation
INITIAL_ROUNDS_MESSAGE_COUNT = 6  # Messages before convergence (orchestrator + specialists × 2 rounds)


def _get_round_purpose(current_round: int, max_rounds: int) -> str:
    """Get the purpose description for the current negotiation round."""
    if current_round == 1:
        return "Independent Proposals - Generate 3-6 hypotheses from your specialist perspective"
    elif current_round == 2:
        return "Cross-Critique - Critique at least 2 hypotheses from other specialists"
    else:
        return "Convergence & Predictions - Converge on strongest hypotheses and generate testable predictions"


def _format_proposals_for_critique(
    geneticist_proposals: List[Hypothesis],
    systems_theorist_proposals: List[Hypothesis],
    predictive_cognition_proposals: List[Hypothesis],
    exclude_role: str
) -> str:
    """Format proposals from other specialists for critique round."""
    sections = []
    
    if exclude_role != "geneticist" and geneticist_proposals:
        proposals_text = "\n".join([
            f"  - [{h.id}] {h.statement} (Confidence: {h.confidence})"
            for h in geneticist_proposals
        ])
        sections.append(f"**Geneticist Proposals:**\n{proposals_text}")
    
    if exclude_role != "systems_theorist" and systems_theorist_proposals:
        proposals_text = "\n".join([
            f"  - [{h.id}] {h.statement} (Confidence: {h.confidence})"
            for h in systems_theorist_proposals
        ])
        sections.append(f"**Systems Theorist Proposals:**\n{proposals_text}")
    
    if exclude_role != "predictive_cognition" and predictive_cognition_proposals:
        proposals_text = "\n".join([
            f"  - [{h.id}] {h.statement} (Confidence: {h.confidence})"
            for h in predictive_cognition_proposals
        ])
        sections.append(f"**Predictive Cognition Scientist Proposals:**\n{proposals_text}")
    
    return "\n\n".join(sections) if sections else "No proposals from other specialists yet."


def _format_all_proposals_and_critiques(
    geneticist_proposals: List[Hypothesis],
    systems_theorist_proposals: List[Hypothesis],
    predictive_cognition_proposals: List[Hypothesis],
    critiques: List[str]
) -> str:
    """Format all proposals and critiques for convergence round."""
    all_proposals = []
    
    for h in geneticist_proposals:
        all_proposals.append(f"[{h.id}] (Geneticist) {h.statement}\n  Rationale: {h.rationale}")
    for h in systems_theorist_proposals:
        all_proposals.append(f"[{h.id}] (Systems Theorist) {h.statement}\n  Rationale: {h.rationale}")
    for h in predictive_cognition_proposals:
        all_proposals.append(f"[{h.id}] (Predictive Cognition) {h.statement}\n  Rationale: {h.rationale}")
    
    critiques_text = "\n".join(critiques) if critiques else "No critiques recorded."
    
    return f"""**All Hypotheses:**
{chr(10).join(all_proposals)}

**Critiques:**
{critiques_text}"""


def _parse_hypotheses_from_response(content: str, role: str, id_prefix: str) -> List[Hypothesis]:
    """Parse hypotheses from a specialist's response text using heuristic pattern matching.
    
    This is a simplified parser that extracts hypothesis-like structures from free-form
    LLM output. It uses pattern matching to identify hypothesis markers (numbered lists,
    headers) and extracts associated metadata from subsequent lines.
    
    Parsing Strategy:
    1. Look for common hypothesis markers (e.g., "1.", "H1", "Hypothesis 1")
    2. Capture the statement from the line containing the marker
    3. Parse subsequent lines for metadata (rationale, assumptions, variables, confidence)
    4. Truncate fields to reasonable lengths to prevent excessively long content
    
    Limitations:
    - May miss hypotheses with non-standard formatting
    - Metadata extraction is keyword-based and may miss nuanced content
    - Field truncation may cut off important information
    
    Note: In production environments with LLMs that reliably support structured output,
    using `model.with_structured_output(Hypothesis)` would be preferred for reliability.
    This fallback parser ensures the feature works even when structured output fails.
    
    Args:
        content: Free-form text response from a specialist agent
        role: The specialist role (e.g., "geneticist", "systems_theorist")
        id_prefix: Prefix for hypothesis IDs (e.g., "G" for geneticist hypotheses)
        
    Returns:
        List of parsed Hypothesis objects, empty list if no hypotheses detected
    """
    hypotheses = []
    
    # Split by common hypothesis markers
    lines = content.split('\n')
    current_hypothesis = None
    hypothesis_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for numbered hypothesis markers
        if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 
                             'Hypothesis 1', 'Hypothesis 2', 'Hypothesis 3', 'Hypothesis 4',
                             '**Hypothesis', '- Hypothesis'))):
            if current_hypothesis:
                hypotheses.append(current_hypothesis)
            
            hypothesis_count += 1
            # Extract the statement (text after the marker)
            statement = line.split(':', 1)[-1].strip() if ':' in line else line
            statement = statement.lstrip('0123456789.-) ').strip()
            
            current_hypothesis = Hypothesis(
                id=f"{id_prefix}{hypothesis_count}",
                statement=statement[:MAX_HYPOTHESIS_STATEMENT_LENGTH],
                rationale="",
                assumptions=[],
                key_variables=[],
                supporting_evidence=[],
                counter_evidence=[],
                confidence="medium",
                proposing_role=role
            )
        elif current_hypothesis:
            # Try to extract additional fields based on keywords
            lower_line = line.lower()
            if 'rationale' in lower_line or 'reasoning' in lower_line:
                current_hypothesis.rationale = line.split(':', 1)[-1].strip()[:MAX_HYPOTHESIS_RATIONALE_LENGTH]
            elif 'assumption' in lower_line:
                current_hypothesis.assumptions.append(line.split(':', 1)[-1].strip()[:MAX_HYPOTHESIS_ASSUMPTION_LENGTH])
            elif 'variable' in lower_line:
                current_hypothesis.key_variables.append(line.split(':', 1)[-1].strip()[:MAX_HYPOTHESIS_VARIABLE_LENGTH])
            elif 'confidence' in lower_line:
                if 'high' in lower_line:
                    current_hypothesis.confidence = 'high'
                elif 'low' in lower_line:
                    current_hypothesis.confidence = 'low'
    
    if current_hypothesis:
        hypotheses.append(current_hypothesis)
    
    return hypotheses


async def negotiation_orchestrator(state: NegotiationState, config: RunnableConfig) -> Dict[str, Any]:
    """Orchestrator node that coordinates the negotiation meeting.
    
    Determines the current round, prepares context for specialists,
    and routes to appropriate round handling.
    """
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", 2)
    
    # Record orchestration decision
    round_purpose = _get_round_purpose(current_round, max_rounds)
    
    orchestration_message = AIMessage(
        content=f"[Orchestrator] Starting Round {current_round}/{max_rounds}: {round_purpose}"
    )
    
    return {
        "negotiation_messages": [orchestration_message],
        "current_round": current_round
    }


async def _run_specialist_agent(
    role: str,
    system_prompt_template: str,
    state: NegotiationState,
    config: RunnableConfig,
    additional_instructions: str
) -> str:
    """Run a specialist agent and return its response content."""
    configurable = Configuration.from_runnable_config(config)
    
    # Use negotiation model or fall back to research model
    model_name = configurable.negotiation_model or configurable.research_model
    model_config = {
        "model": model_name,
        "max_tokens": configurable.negotiation_max_tokens,
        "api_key": get_api_key_for_model(model_name, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare notes context
    notes = state.get("notes", [])
    raw_notes = state.get("raw_notes", [])
    notes_context = "\n".join(notes + raw_notes)[:10000]  # Limit context size
    
    # Format system prompt
    system_prompt = system_prompt_template.format(
        research_brief=state.get("research_brief", ""),
        notes=notes_context,
        current_round=state.get("current_round", 1),
        max_rounds=state.get("max_rounds", 2),
        round_purpose=_get_round_purpose(state.get("current_round", 1), state.get("max_rounds", 2)),
        additional_instructions=additional_instructions
    )
    
    model = configurable_model.with_config(model_config)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Please provide your specialist contribution for this round.")
    ]
    
    response = await model.ainvoke(messages)
    return str(response.content)


async def geneticist_agent(state: NegotiationState, config: RunnableConfig) -> Dict[str, Any]:
    """Geneticist specialist agent node."""
    current_round = state.get("current_round", 1)
    
    # Prepare additional instructions based on round
    if current_round == 1:
        additional_instructions = specialist_proposal_round_instructions
    elif current_round == 2:
        other_proposals = _format_proposals_for_critique(
            state.get("geneticist_proposals", []),
            state.get("systems_theorist_proposals", []),
            state.get("predictive_cognition_proposals", []),
            exclude_role="geneticist"
        )
        additional_instructions = specialist_critique_round_instructions.format(
            other_proposals=other_proposals
        )
    else:
        proposals_and_critiques = _format_all_proposals_and_critiques(
            state.get("geneticist_proposals", []),
            state.get("systems_theorist_proposals", []),
            state.get("predictive_cognition_proposals", []),
            state.get("critiques", [])
        )
        additional_instructions = specialist_convergence_instructions.format(
            proposals_and_critiques=proposals_and_critiques
        )
    
    content = await _run_specialist_agent(
        role="geneticist",
        system_prompt_template=geneticist_system_prompt,
        state=state,
        config=config,
        additional_instructions=additional_instructions
    )
    
    result: Dict[str, Any] = {
        "negotiation_messages": [AIMessage(content=f"[Geneticist] {content}")]
    }
    
    # Parse proposals in round 1
    if current_round == 1:
        proposals = _parse_hypotheses_from_response(content, "geneticist", "G")
        result["geneticist_proposals"] = proposals
    elif current_round == 2:
        # Store critiques
        result["critiques"] = [f"[Geneticist Critique] {content}"]
    
    return result


async def systems_theorist_agent(state: NegotiationState, config: RunnableConfig) -> Dict[str, Any]:
    """Systems Theorist specialist agent node."""
    current_round = state.get("current_round", 1)
    
    # Prepare additional instructions based on round
    if current_round == 1:
        additional_instructions = specialist_proposal_round_instructions
    elif current_round == 2:
        other_proposals = _format_proposals_for_critique(
            state.get("geneticist_proposals", []),
            state.get("systems_theorist_proposals", []),
            state.get("predictive_cognition_proposals", []),
            exclude_role="systems_theorist"
        )
        additional_instructions = specialist_critique_round_instructions.format(
            other_proposals=other_proposals
        )
    else:
        proposals_and_critiques = _format_all_proposals_and_critiques(
            state.get("geneticist_proposals", []),
            state.get("systems_theorist_proposals", []),
            state.get("predictive_cognition_proposals", []),
            state.get("critiques", [])
        )
        additional_instructions = specialist_convergence_instructions.format(
            proposals_and_critiques=proposals_and_critiques
        )
    
    content = await _run_specialist_agent(
        role="systems_theorist",
        system_prompt_template=systems_theorist_system_prompt,
        state=state,
        config=config,
        additional_instructions=additional_instructions
    )
    
    result: Dict[str, Any] = {
        "negotiation_messages": [AIMessage(content=f"[Systems Theorist] {content}")]
    }
    
    # Parse proposals in round 1
    if current_round == 1:
        proposals = _parse_hypotheses_from_response(content, "systems_theorist", "S")
        result["systems_theorist_proposals"] = proposals
    elif current_round == 2:
        result["critiques"] = [f"[Systems Theorist Critique] {content}"]
    
    return result


async def predictive_cognition_agent(state: NegotiationState, config: RunnableConfig) -> Dict[str, Any]:
    """Predictive Cognition Scientist specialist agent node."""
    current_round = state.get("current_round", 1)
    
    # Prepare additional instructions based on round
    if current_round == 1:
        additional_instructions = specialist_proposal_round_instructions
    elif current_round == 2:
        other_proposals = _format_proposals_for_critique(
            state.get("geneticist_proposals", []),
            state.get("systems_theorist_proposals", []),
            state.get("predictive_cognition_proposals", []),
            exclude_role="predictive_cognition"
        )
        additional_instructions = specialist_critique_round_instructions.format(
            other_proposals=other_proposals
        )
    else:
        proposals_and_critiques = _format_all_proposals_and_critiques(
            state.get("geneticist_proposals", []),
            state.get("systems_theorist_proposals", []),
            state.get("predictive_cognition_proposals", []),
            state.get("critiques", [])
        )
        additional_instructions = specialist_convergence_instructions.format(
            proposals_and_critiques=proposals_and_critiques
        )
    
    content = await _run_specialist_agent(
        role="predictive_cognition",
        system_prompt_template=predictive_cognition_system_prompt,
        state=state,
        config=config,
        additional_instructions=additional_instructions
    )
    
    result: Dict[str, Any] = {
        "negotiation_messages": [AIMessage(content=f"[Predictive Cognition Scientist] {content}")]
    }
    
    # Parse proposals in round 1
    if current_round == 1:
        proposals = _parse_hypotheses_from_response(content, "predictive_cognition", "P")
        result["predictive_cognition_proposals"] = proposals
    elif current_round == 2:
        result["critiques"] = [f"[Predictive Cognition Critique] {content}"]
    
    return result


async def negotiation_round_router(state: NegotiationState, config: RunnableConfig) -> Dict[str, Any]:
    """Route to next round or synthesis based on current state."""
    current_round = state.get("current_round", 1)
    
    # Increment round counter for next iteration
    return {"current_round": current_round + 1}


def should_continue_negotiation(state: NegotiationState) -> Literal["specialists", "synthesis"]:
    """Determine if negotiation should continue or proceed to synthesis."""
    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", 2)
    
    if current_round <= max_rounds:
        return "specialists"
    return "synthesis"


async def negotiation_synthesis(state: NegotiationState, config: RunnableConfig) -> Dict[str, Any]:
    """Synthesize all negotiation outputs into a structured HypothesesBundle."""
    configurable = Configuration.from_runnable_config(config)
    
    # Use negotiation model or fall back to research model
    model_name = configurable.negotiation_model or configurable.research_model
    model_config = {
        "model": model_name,
        "max_tokens": configurable.negotiation_max_tokens,
        "api_key": get_api_key_for_model(model_name, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Gather all proposals
    all_hypotheses = []
    for h in state.get("geneticist_proposals", []):
        all_hypotheses.append(f"[{h.id}] (Geneticist) {h.statement}\n  Rationale: {h.rationale}")
    for h in state.get("systems_theorist_proposals", []):
        all_hypotheses.append(f"[{h.id}] (Systems Theorist) {h.statement}\n  Rationale: {h.rationale}")
    for h in state.get("predictive_cognition_proposals", []):
        all_hypotheses.append(f"[{h.id}] (Predictive Cognition) {h.statement}\n  Rationale: {h.rationale}")
    
    all_hypotheses_text = "\n\n".join(all_hypotheses)
    critiques_text = "\n\n".join(state.get("critiques", []))
    
    # Extract convergence notes from messages after initial proposal and critique rounds
    # INITIAL_ROUNDS_MESSAGE_COUNT accounts for orchestrator + specialists in first 2 rounds
    messages = state.get("negotiation_messages", [])
    convergence_notes = ""
    if len(messages) > INITIAL_ROUNDS_MESSAGE_COUNT:
        convergence_notes = "\n".join([
            str(m.content) for m in messages[INITIAL_ROUNDS_MESSAGE_COUNT:]
            if hasattr(m, 'content')
        ])[:5000]
    
    # Prepare synthesis prompt
    synthesis_prompt = negotiation_synthesis_prompt.format(
        research_brief=state.get("research_brief", ""),
        all_hypotheses=all_hypotheses_text,
        all_critiques=critiques_text,
        convergence_notes=convergence_notes
    )
    
    model = configurable_model.with_config(model_config)
    
    try:
        # Try to get structured output
        structured_model = model.with_structured_output(HypothesesBundle).with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        )
        
        response = await structured_model.ainvoke([
            SystemMessage(content="You are synthesizing scientific negotiation outputs."),
            HumanMessage(content=synthesis_prompt)
        ])
        
        hypotheses_bundle = response
        
    except Exception:
        # Fall back to manual bundle construction if structured output fails
        all_proposals = (
            state.get("geneticist_proposals", []) +
            state.get("systems_theorist_proposals", []) +
            state.get("predictive_cognition_proposals", [])
        )
        
        # Create a basic bundle from collected proposals, limiting to top hypotheses
        hypotheses_bundle = HypothesesBundle(
            hypotheses=all_proposals[:MAX_FALLBACK_HYPOTHESES],
            predictions=[],
            open_questions=[
                "Further research needed to validate hypotheses",
                "Cross-disciplinary experiments recommended"
            ],
            disagreements=[]
        )
    
    synthesis_message = AIMessage(
        content=f"[Synthesis Complete] Generated {len(hypotheses_bundle.hypotheses)} hypotheses, "
                f"{len(hypotheses_bundle.predictions)} predictions, "
                f"{len(hypotheses_bundle.disagreements)} documented disagreements."
    )
    
    return {
        "hypotheses_bundle": hypotheses_bundle,
        "negotiation_messages": [synthesis_message]
    }


# Build the negotiation subgraph
negotiation_builder = StateGraph(NegotiationState, config_schema=Configuration)

# Add nodes
negotiation_builder.add_node("orchestrator", negotiation_orchestrator)
negotiation_builder.add_node("geneticist", geneticist_agent)
negotiation_builder.add_node("systems_theorist", systems_theorist_agent)
negotiation_builder.add_node("predictive_cognition", predictive_cognition_agent)
negotiation_builder.add_node("round_router", negotiation_round_router)
negotiation_builder.add_node("synthesis", negotiation_synthesis)

# Define edges
# Entry: orchestrator coordinates the round
negotiation_builder.add_edge(START, "orchestrator")

# Orchestrator -> all specialists run in parallel
negotiation_builder.add_edge("orchestrator", "geneticist")
negotiation_builder.add_edge("orchestrator", "systems_theorist")
negotiation_builder.add_edge("orchestrator", "predictive_cognition")

# All specialists -> round router (join point)
negotiation_builder.add_edge("geneticist", "round_router")
negotiation_builder.add_edge("systems_theorist", "round_router")
negotiation_builder.add_edge("predictive_cognition", "round_router")

# Round router -> conditional edge: continue or synthesize
negotiation_builder.add_conditional_edges(
    "round_router",
    should_continue_negotiation,
    {
        "specialists": "orchestrator",  # Loop back for next round
        "synthesis": "synthesis"         # Proceed to synthesis
    }
)

# Synthesis -> END
negotiation_builder.add_edge("synthesis", END)

# Compile the negotiation subgraph
negotiation_subgraph = negotiation_builder.compile()


async def scientific_negotiation(state: AgentState, config: RunnableConfig) -> Command[Literal["final_report_generation"]]:
    """Scientific negotiation node for the main workflow.
    
    This node runs the multi-round specialist meeting if enabled in configuration.
    If disabled, it skips directly to final report generation.
    
    Args:
        state: Current agent state with research findings
        config: Runtime configuration with negotiation settings
        
    Returns:
        Command to proceed to final report generation, optionally with hypotheses bundle
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Skip negotiation if not enabled
    if not configurable.enable_scientific_negotiation:
        return Command(goto="final_report_generation")
    
    # Prepare initial state for negotiation subgraph
    notes = state.get("notes", [])
    raw_notes = state.get("raw_notes", [])
    research_brief = state.get("research_brief", "")
    
    negotiation_input = {
        "research_brief": research_brief,
        "notes": notes,
        "raw_notes": raw_notes,
        "current_round": 1,
        "max_rounds": configurable.negotiation_rounds,
        "geneticist_proposals": [],
        "systems_theorist_proposals": [],
        "predictive_cognition_proposals": [],
        "critiques": [],
        "hypotheses_bundle": None,
        "negotiation_messages": []
    }
    
    # Run the negotiation subgraph
    try:
        result = await negotiation_subgraph.ainvoke(negotiation_input, config)
        
        return Command(
            goto="final_report_generation",
            update={
                "hypotheses_bundle": result.get("hypotheses_bundle"),
                "negotiation_messages": result.get("negotiation_messages", [])
            }
        )
    except Exception as e:
        # If negotiation fails, continue without it
        error_message = AIMessage(
            content=f"[Scientific Negotiation] Failed to complete negotiation: {str(e)}"
        )
        return Command(
            goto="final_report_generation",
            update={"negotiation_messages": [error_message]}
        )


def _format_hypotheses_bundle_for_report(bundle: HypothesesBundle) -> str:
    """Format the hypotheses bundle as a section for the final report.
    
    Args:
        bundle: The HypothesesBundle from scientific negotiation
        
    Returns:
        Formatted markdown string for inclusion in the report
    """
    sections = ["## Scientific Negotiation Results\n"]
    sections.append("The following hypotheses were generated through a multi-round scientific negotiation between specialist agents (Geneticist, Systems Theorist, and Predictive Cognition Scientist).\n")
    
    # Hypotheses section
    if bundle.hypotheses:
        sections.append("### Generated Hypotheses\n")
        for h in bundle.hypotheses:
            sections.append(f"**{h.id}: {h.statement}**")
            sections.append(f"- *Rationale*: {h.rationale}")
            if h.assumptions:
                sections.append(f"- *Assumptions*: {', '.join(h.assumptions)}")
            if h.key_variables:
                sections.append(f"- *Key Variables*: {', '.join(h.key_variables)}")
            if h.supporting_evidence:
                sections.append(f"- *Supporting Evidence*: {'; '.join(h.supporting_evidence)}")
            if h.counter_evidence:
                sections.append(f"- *Counter Evidence*: {'; '.join(h.counter_evidence)}")
            sections.append(f"- *Confidence*: {h.confidence}")
            if h.proposing_role:
                sections.append(f"- *Proposed by*: {h.proposing_role}")
            sections.append("")
    
    # Predictions section
    if bundle.predictions:
        sections.append("### Testable Predictions\n")
        for i, p in enumerate(bundle.predictions, 1):
            sections.append(f"**Prediction {i}**: {p.prediction}")
            sections.append(f"- *Related Hypotheses*: {', '.join(p.hypothesis_ids)}")
            sections.append(f"- *Test Method*: {p.test_method}")
            if p.required_data:
                sections.append(f"- *Required Data*: {', '.join(p.required_data)}")
            if p.expected_if_true:
                sections.append(f"- *Expected if True*: {p.expected_if_true}")
            if p.expected_if_false:
                sections.append(f"- *Expected if False*: {p.expected_if_false}")
            sections.append("")
    
    # Open questions section
    if bundle.open_questions:
        sections.append("### Open Questions\n")
        for q in bundle.open_questions:
            sections.append(f"- {q}")
        sections.append("")
    
    # Disagreements section
    if bundle.disagreements:
        sections.append("### Unresolved Disagreements\n")
        for d in bundle.disagreements:
            sections.append(f"**Topic**: {d.topic}")
            for role, position in d.positions_by_role.items():
                sections.append(f"- *{role}*: {position}")
            if d.what_data_would_resolve:
                sections.append(f"- *Resolution Data*: {d.what_data_would_resolve}")
            sections.append("")
    
    return "\n".join(sections)

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    If a hypotheses_bundle exists from scientific negotiation, it is included in the report.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 1b: Include hypotheses bundle if present
    hypotheses_bundle = state.get("hypotheses_bundle")
    hypotheses_section = ""
    if hypotheses_bundle:
        hypotheses_section = _format_hypotheses_bundle_for_report(hypotheses_bundle)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            # Include hypotheses bundle section if available
            enhanced_findings = findings
            if hypotheses_section:
                enhanced_findings = f"{findings}\n\n{hypotheses_section}"
            
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=enhanced_findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # Return successful report generation
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("scientific_negotiation", scientific_negotiation) # Scientific negotiation phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "scientific_negotiation")  # Research to negotiation
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()