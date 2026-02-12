"""Main LangGraph implementation for the Deep Research agent.

Architecture: Forum-based multi-agent coordination.

The graph is structured as a **forum** — a full, nondirected, frequently
cyclical graph — rather than a simple chain.  The human supervisor acts
as the central hub with conditional, interruptible edges to every other
agent node.  All interactions are recorded in a timestamped ledger and
presented as a play-script transcript so that every participant's
contributions are visible at all stages.
"""

import asyncio
from typing import Any, Dict, List, Literal, Optional

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
from langgraph.types import Command, interrupt

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    geneticist_system_prompt,
    human_directive_classification_prompt,
    human_supervisor_status_prompt,
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
    ConductNegotiationRound,
    ConductResearch,
    HumanDirective,
    Hypothesis,
    HypothesesBundle,
    LedgerEntry,
    NegotiationState,
    QuerySpecialist,
    RecallFromNegotiation,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorSpecialistQuery,
    SupervisorState,
    SynthesizeNegotiation,
    format_ledger_entry_as_script_line,
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

# Maximum length for notes context in specialist queries (token budget constraint)
MAX_NOTES_CONTEXT_LENGTH = 10000

# Negotiation-related constants
MAX_RECALL_MESSAGES = 50  # Max messages to include in recall context (token budget constraint)
MAX_PREVIEW_MESSAGES = 10  # Max messages to preview in negotiation round tool response


def _make_ledger_update(
    agent: str,
    action: str,
    content: str,
    target: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create a ledger entry dict and its play-script transcript line.

    Returns a dict suitable for merging into a ``Command.update`` payload::

        {
            "ledger": [<serialised LedgerEntry>],
            "forum_transcript": ["[ts] AGENT → TARGET: (action) content"],
        }
    """
    entry = LedgerEntry(
        agent=agent,
        action=action,
        content=content[:500],  # Truncate to prevent unbounded growth
        target=target,
        metadata=metadata or {},
    )
    return {
        "ledger": [entry.model_dump()],
        "forum_transcript": [format_ledger_entry_as_script_line(entry)],
    }

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
        ledger_update = _make_ledger_update(
            agent="research_supervisor",
            action="clarification_request",
            content=response.question,
            target="human_supervisor",
        )
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)], **ledger_update}
        )
    else:
        # Proceed to research with verification message
        ledger_update = _make_ledger_update(
            agent="research_supervisor",
            action="clarification_resolved",
            content=response.verification,
        )
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)], **ledger_update}
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
    
    ledger_update = _make_ledger_update(
        agent="research_supervisor",
        action="research_brief_created",
        content=response.research_brief,
        target="research_supervisor",
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
            },
            # Initialize negotiation state fields
            "negotiation_round": 0,
            "negotiation_max_rounds": configurable.negotiation_rounds,
            "negotiation_messages": [],
            "geneticist_proposals": [],
            "systems_theorist_proposals": [],
            "predictive_cognition_proposals": [],
            "critiques": [],
            "hypotheses_bundle": None,
            **ledger_update,
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
    notes_context = "\n".join(notes + raw_notes)[:MAX_NOTES_CONTEXT_LENGTH]
    
    # Step 3: Sanitize question to prevent prompt injection
    # Replace XML-like tags and special characters that could interfere with prompt structure
    sanitized_question = question.replace("<", "&lt;").replace(">", "&gt;")
    
    # Step 4: Format prompt for direct consultation
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

Question: {sanitized_question}
</Supervisor Direct Query>
"""
    )
    
    # Step 5: Configure and invoke specialist model
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


async def synthesize_negotiation(
    state: SupervisorState,
    config: RunnableConfig,
    synthesis_instructions: str
) -> HypothesesBundle:
    """Synthesize all negotiation outputs into a structured HypothesesBundle.
    
    This function takes all accumulated proposals and critiques from negotiation rounds
    and synthesizes them into a comprehensive hypotheses bundle with predictions,
    open questions, and documented disagreements.
    
    Args:
        state: Current supervisor state with all negotiation data
        config: Runtime configuration
        synthesis_instructions: Instructions for synthesis process
        
    Returns:
        HypothesesBundle with synthesized hypotheses and predictions
    """
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
    
    all_hypotheses_text = "\n\n".join(all_hypotheses) if all_hypotheses else "No hypotheses proposed yet"
    critiques_text = "\n\n".join(state.get("critiques", [])) if state.get("critiques") else "No critiques provided yet"
    
    # Extract convergence notes from negotiation messages
    messages = state.get("negotiation_messages", [])
    convergence_notes = ""
    if messages:
        # Get recent messages (skip initial orchestrator messages)
        recent_messages = [
            str(m.content) for m in messages[-20:] if hasattr(m, 'content')
        ]
        convergence_notes = "\n".join(recent_messages)[:5000]
    
    # Prepare synthesis prompt
    synthesis_prompt = negotiation_synthesis_prompt.format(
        research_brief=state.get("research_brief", ""),
        all_hypotheses=all_hypotheses_text,
        all_critiques=critiques_text,
        convergence_notes=convergence_notes
    )
    
    # Add custom synthesis instructions
    full_prompt = f"{synthesis_prompt}\n\nAdditional Synthesis Instructions:\n{synthesis_instructions}"
    
    model = configurable_model.with_config(model_config)
    
    try:
        # Try to get structured output
        structured_model = model.with_structured_output(HypothesesBundle).with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        )
        
        response = await structured_model.ainvoke([
            SystemMessage(content="You are synthesizing scientific negotiation outputs."),
            HumanMessage(content=full_prompt)
        ])
        
        return response
        
    except Exception:
        # Fall back to manual bundle construction if structured output fails
        all_proposals = (
            state.get("geneticist_proposals", []) +
            state.get("systems_theorist_proposals", []) +
            state.get("predictive_cognition_proposals", [])
        )
        
        # Create a basic bundle from collected proposals, limiting to top hypotheses
        return HypothesesBundle(
            hypotheses=all_proposals[:MAX_FALLBACK_HYPOTHESES],
            predictions=[],
            open_questions=[
                "Further research needed to validate hypotheses",
                "Cross-disciplinary experiments recommended"
            ],
            disagreements=[]
        )


async def conduct_single_negotiation_round(
    state: SupervisorState, 
    config: RunnableConfig, 
    round_instructions: str
) -> Dict[str, Any]:
    """Execute a single round of negotiation (orchestrator + specialists).
    
    This function runs one iteration of the negotiation cycle where:
    1. The orchestrator prepares the round
    2. All three specialists contribute in parallel
    3. Results are collected and returned to the supervisor
    
    Args:
        state: Current supervisor state with negotiation context
        config: Runtime configuration
        round_instructions: High-level instructions for this specific round
        
    Returns:
        Dictionary with updated negotiation state fields
    """
    configurable = Configuration.from_runnable_config(config)
    current_round = state.get("negotiation_round", 0) + 1
    max_rounds = state.get("negotiation_max_rounds", 2)
    
    # Create NegotiationState from SupervisorState
    negotiation_state: NegotiationState = {
        "research_brief": state.get("research_brief", ""),
        "notes": state.get("notes", []),
        "raw_notes": state.get("raw_notes", []),
        "current_round": current_round,
        "max_rounds": max_rounds,
        "geneticist_proposals": state.get("geneticist_proposals", []),
        "systems_theorist_proposals": state.get("systems_theorist_proposals", []),
        "predictive_cognition_proposals": state.get("predictive_cognition_proposals", []),
        "critiques": state.get("critiques", []),
        "hypotheses_bundle": state.get("hypotheses_bundle"),
        "negotiation_messages": state.get("negotiation_messages", [])
    }
    
    # Run orchestrator
    orchestrator_result = await negotiation_orchestrator(negotiation_state, config)
    negotiation_state["negotiation_messages"] = negotiation_state["negotiation_messages"] + orchestrator_result.get("negotiation_messages", [])
    
    # Add custom round instructions to messages
    custom_instruction_message = AIMessage(
        content=f"[Supervisor Instructions for Round {current_round}] {round_instructions}"
    )
    negotiation_state["negotiation_messages"].append(custom_instruction_message)
    
    # Run all three specialists in parallel
    geneticist_task = geneticist_agent(negotiation_state, config)
    systems_task = systems_theorist_agent(negotiation_state, config)
    predictive_task = predictive_cognition_agent(negotiation_state, config)
    
    results = await asyncio.gather(geneticist_task, systems_task, predictive_task)
    
    # Collect results
    update_dict: Dict[str, Any] = {
        "negotiation_round": current_round,
        "negotiation_messages": [],
        "critiques": []  # Initialize critiques list
    }
    
    for result in results:
        if "negotiation_messages" in result:
            update_dict["negotiation_messages"].extend(result["negotiation_messages"])
        if "geneticist_proposals" in result:
            update_dict["geneticist_proposals"] = result["geneticist_proposals"]
        if "systems_theorist_proposals" in result:
            update_dict["systems_theorist_proposals"] = result["systems_theorist_proposals"]
        if "predictive_cognition_proposals" in result:
            update_dict["predictive_cognition_proposals"] = result["predictive_cognition_proposals"]
        if "critiques" in result:
            update_dict["critiques"].extend(result["critiques"])
    
    return update_dict


async def recall_from_negotiation(
    state: SupervisorState,
    config: RunnableConfig,
    query: str,
    specialist_filter: Optional[str] = None
) -> str:
    """Search negotiation conversation history for specific information.
    
    This implements RAG-like recall over the accumulated negotiation_messages,
    optionally filtered by specialist role.
    
    Args:
        state: Current supervisor state with negotiation history
        config: Runtime configuration
        query: What to search for in the conversation history
        specialist_filter: Optional specialist role to filter by
        
    Returns:
        Formatted string with relevant excerpts from negotiation history
    """
    negotiation_messages = state.get("negotiation_messages", [])
    
    if not negotiation_messages:
        return "No negotiation history available yet. Start negotiation rounds first."
    
    # Filter messages by specialist if requested
    relevant_messages = []
    for msg in negotiation_messages:
        if not hasattr(msg, 'content'):
            continue
        
        content = str(msg.content)
        
        # Apply specialist filter
        if specialist_filter:
            specialist_name = specialist_filter.replace("_", " ").title()
            if f"[{specialist_name}" not in content and f"[{specialist_filter}" not in content.lower():
                continue
        
        relevant_messages.append(content)
    
    if not relevant_messages:
        filter_msg = f" from {specialist_filter}" if specialist_filter else ""
        return f"No relevant messages found{filter_msg} in negotiation history."
    
    # Use LLM to extract relevant information based on query
    configurable = Configuration.from_runnable_config(config)
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    model = configurable_model.with_config(model_config)
    
    # Combine messages for context
    # Note: Older messages beyond MAX_RECALL_MESSAGES are dropped to prevent token overflow.
    # This prioritizes recent negotiation history which is typically most relevant for recall queries.
    messages_context = "\n\n".join(relevant_messages[:MAX_RECALL_MESSAGES])
    
    recall_prompt = f"""You are helping the Research Supervisor recall specific information from the scientific negotiation conversation history.

<Query>
{query}
</Query>

<Negotiation History>
{messages_context}
</Negotiation History>

Instructions:
1. Extract and summarize information relevant to the query
2. Cite which specialist(s) provided the information
3. Keep the response focused and concise
4. If the information isn't in the history, clearly state that

Provide your response:"""
    
    response = await model.ainvoke([
        SystemMessage(content="You are a research assistant helping to recall information from conversation history."),
        HumanMessage(content=recall_prompt)
    ])
    
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
    
    # Available tools: research delegation, completion signaling, specialist queries, negotiation control, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, QuerySpecialist, ConductNegotiationRound, RecallFromNegotiation, SynthesizeNegotiation, think_tool]
    
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
                "research_brief": state.get("research_brief", ""),
                # Pass negotiation state to AgentState
                "negotiation_round": state.get("negotiation_round", 0),
                "negotiation_max_rounds": state.get("negotiation_max_rounds", 2),
                "negotiation_messages": state.get("negotiation_messages", []),
                "geneticist_proposals": state.get("geneticist_proposals", []),
                "systems_theorist_proposals": state.get("systems_theorist_proposals", []),
                "predictive_cognition_proposals": state.get("predictive_cognition_proposals", []),
                "critiques": state.get("critiques", []),
                "hypotheses_bundle": state.get("hypotheses_bundle")
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
    
    # Handle ConductNegotiationRound calls (negotiation round execution)
    conduct_negotiation_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductNegotiationRound"
    ]
    
    if conduct_negotiation_calls:
        for tool_call in conduct_negotiation_calls:
            round_instructions = tool_call["args"]["round_instructions"]
            
            try:
                # Execute single negotiation round
                round_result = await conduct_single_negotiation_round(
                    state=state,
                    config=config,
                    round_instructions=round_instructions
                )
                
                # Format response message
                current_round = round_result.get("negotiation_round", state.get("negotiation_round", 0))
                num_messages = len(round_result.get("negotiation_messages", []))
                
                response_content = f"[Negotiation Round {current_round} Complete]\n"
                response_content += f"Instructions: {round_instructions}\n"
                response_content += f"Generated {num_messages} specialist contributions.\n\n"
                
                # Include summary of specialist messages
                for msg in round_result.get("negotiation_messages", [])[:MAX_PREVIEW_MESSAGES]:  # Limit for brevity
                    if hasattr(msg, 'content'):
                        content = str(msg.content)[:300]  # Truncate long messages
                        response_content += f"{content}\n\n"
                
                all_tool_messages.append(ToolMessage(
                    content=response_content,
                    name="ConductNegotiationRound",
                    tool_call_id=tool_call["id"]
                ))
                
                # Update state with negotiation results
                for key in ["negotiation_round", "negotiation_messages", "geneticist_proposals", 
                           "systems_theorist_proposals", "predictive_cognition_proposals", "critiques"]:
                    if key in round_result:
                        update_payload[key] = round_result[key]
                
            except Exception as e:
                error_msg = f"Error conducting negotiation round: {str(e)}"
                all_tool_messages.append(ToolMessage(
                    content=error_msg,
                    name="ConductNegotiationRound",
                    tool_call_id=tool_call["id"]
                ))
    
    # Handle RecallFromNegotiation calls (conversation history recall)
    recall_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "RecallFromNegotiation"
    ]
    
    if recall_calls:
        for tool_call in recall_calls:
            query = tool_call["args"]["query"]
            specialist_filter = tool_call["args"].get("specialist_filter")
            
            try:
                # Execute recall
                recall_result = await recall_from_negotiation(
                    state=state,
                    config=config,
                    query=query,
                    specialist_filter=specialist_filter
                )
                
                filter_text = f" (filtered to {specialist_filter})" if specialist_filter else ""
                response_content = f"[Recall Result{filter_text}]\nQuery: {query}\n\n{recall_result}"
                
                all_tool_messages.append(ToolMessage(
                    content=response_content,
                    name="RecallFromNegotiation",
                    tool_call_id=tool_call["id"]
                ))
                
            except Exception as e:
                error_msg = f"Error recalling from negotiation: {str(e)}"
                all_tool_messages.append(ToolMessage(
                    content=error_msg,
                    name="RecallFromNegotiation",
                    tool_call_id=tool_call["id"]
                ))
    
    # Handle SynthesizeNegotiation calls (create final hypotheses bundle)
    synthesize_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "SynthesizeNegotiation"
    ]
    
    if synthesize_calls:
        for tool_call in synthesize_calls:
            synthesis_instructions = tool_call["args"].get("synthesis_instructions", 
                "Synthesize all proposals and critiques into a comprehensive hypotheses bundle")
            
            try:
                # Execute synthesis
                hypotheses_bundle = await synthesize_negotiation(
                    state=state,
                    config=config,
                    synthesis_instructions=synthesis_instructions
                )
                
                # Format response
                num_hypotheses = len(hypotheses_bundle.hypotheses)
                num_predictions = len(hypotheses_bundle.predictions)
                num_disagreements = len(hypotheses_bundle.disagreements)
                
                response_content = f"[Synthesis Complete]\n"
                response_content += f"Generated {num_hypotheses} hypotheses, {num_predictions} predictions, "
                response_content += f"{num_disagreements} documented disagreements.\n\n"
                response_content += f"Hypotheses: {', '.join([h.id for h in hypotheses_bundle.hypotheses])}\n"
                
                all_tool_messages.append(ToolMessage(
                    content=response_content,
                    name="SynthesizeNegotiation",
                    tool_call_id=tool_call["id"]
                ))
                
                # Update state with synthesized bundle
                update_payload["hypotheses_bundle"] = hypotheses_bundle
                
            except Exception as e:
                error_msg = f"Error synthesizing negotiation: {str(e)}"
                all_tool_messages.append(ToolMessage(
                    content=error_msg,
                    name="SynthesizeNegotiation",
                    tool_call_id=tool_call["id"]
                ))
    
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
                        "research_brief": state.get("research_brief", ""),
                        # Pass negotiation state to AgentState
                        "negotiation_round": state.get("negotiation_round", 0),
                        "negotiation_max_rounds": state.get("negotiation_max_rounds", 2),
                        "negotiation_messages": state.get("negotiation_messages", []),
                        "geneticist_proposals": state.get("geneticist_proposals", []),
                        "systems_theorist_proposals": state.get("systems_theorist_proposals", []),
                        "predictive_cognition_proposals": state.get("predictive_cognition_proposals", []),
                        "critiques": state.get("critiques", []),
                        "hypotheses_bundle": state.get("hypotheses_bundle")
                    }
                )
    
    # Step 3: Return command with all tool results and ledger entries
    # Log each tool result to the forum ledger
    ledger_entries: list[dict] = []
    transcript_lines: list[str] = []
    for msg in all_tool_messages:
        entry = LedgerEntry(
            agent="research_supervisor",
            action=f"tool_result:{msg.name}",
            content=str(msg.content)[:500],
            target="research_supervisor",
        )
        ledger_entries.append(entry.model_dump())
        transcript_lines.append(format_ledger_entry_as_script_line(entry))

    update_payload["supervisor_messages"] = all_tool_messages
    if "ledger" not in update_payload:
        update_payload["ledger"] = []
    update_payload["ledger"].extend(ledger_entries)
    if "forum_transcript" not in update_payload:
        update_payload["forum_transcript"] = []
    update_payload["forum_transcript"].extend(transcript_lines)
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
            ledger_update = _make_ledger_update(
                agent="research_supervisor",
                action="final_report_generated",
                content="Final report successfully generated.",
                target="human_supervisor",
            )
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state,
                **ledger_update,
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


async def _classify_human_directive(
    human_message: str,
    config: RunnableConfig
) -> HumanDirective:
    """Classify a free-form human message into a structured HumanDirective.

    Uses an LLM to interpret the human supervisor's instruction and map it
    to a structured action that the system can execute.

    Args:
        human_message: Free-form instruction from the human supervisor
        config: Runtime configuration with model settings

    Returns:
        HumanDirective with classified action, content, and optional specialist
    """
    configurable = Configuration.from_runnable_config(config)
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    classification_model = (
        configurable_model
        .with_structured_output(HumanDirective)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )

    prompt = human_directive_classification_prompt.format(
        human_message=human_message,
    )
    return await classification_model.ainvoke([HumanMessage(content=prompt)])


def _build_status_message(state: AgentState) -> str:
    """Build a research status summary for the human supervisor.

    Args:
        state: Current agent state with research data

    Returns:
        Formatted status string presented to the human
    """
    notes = state.get("notes", [])
    raw_notes = state.get("raw_notes", [])
    all_notes = notes + raw_notes

    if all_notes:
        findings_summary = "\n".join(
            f"- {note[:200]}..." if len(note) > 200 else f"- {note}"
            for note in all_notes[:10]
        )
        if len(all_notes) > 10:
            findings_summary += f"\n... and {len(all_notes) - 10} more notes"
    else:
        findings_summary = "No research notes collected yet."

    has_bundle = "Yes" if state.get("hypotheses_bundle") else "No"

    # Include recent forum transcript lines for play-script visibility
    transcript = state.get("forum_transcript", [])
    if transcript:
        recent_lines = transcript[-20:]  # Show last 20 transcript lines
        transcript_section = "\n".join(recent_lines)
        if len(transcript) > 20:
            transcript_section = f"... ({len(transcript) - 20} earlier entries omitted)\n{transcript_section}"
    else:
        transcript_section = "No forum activity recorded yet."

    return human_supervisor_status_prompt.format(
        research_brief=state.get("research_brief", "Not yet defined"),
        num_notes=len(all_notes),
        findings_summary=findings_summary,
        negotiation_round=state.get("negotiation_round", 0),
        negotiation_max_rounds=state.get("negotiation_max_rounds", 2),
        num_geneticist_proposals=len(state.get("geneticist_proposals", [])),
        num_systems_theorist_proposals=len(state.get("systems_theorist_proposals", [])),
        num_predictive_cognition_proposals=len(state.get("predictive_cognition_proposals", [])),
        num_critiques=len(state.get("critiques", [])),
        has_hypotheses_bundle=has_bundle,
        forum_transcript=transcript_section,
    )


async def human_supervisor(
    state: AgentState, config: RunnableConfig,
) -> Command[Literal["process_human_directive", "final_report_generation", "research_supervisor"]]:
    """Present research status and wait for human supervisor input.

    In the forum architecture the human supervisor has a conditional edge
    to *every* other node, making it the central hub.  The human can
    choose to route control to:
    - ``process_human_directive`` — for research, specialist queries,
      negotiation rounds, recall, synthesis, or feedback
    - ``final_report_generation`` — when satisfied with findings
    - ``research_supervisor`` — to kick off another automated research
      cycle directly

    This node uses ``interrupt()`` to pause execution and present the
    current research state (including the forum transcript / play-script
    view) to the human user via the LangGraph Studio UI.

    Args:
        state: Current agent state
        config: Runtime configuration

    Returns:
        Command routing to process_human_directive or final_report_generation
    """
    status_message = _build_status_message(state)

    # Interrupt and wait for human input, showing the status message
    human_input = interrupt(status_message)

    # If the human provides input, classify it and route accordingly
    if isinstance(human_input, str) and human_input.strip():
        directive = await _classify_human_directive(human_input, config)

        # Log the human directive to the forum ledger
        ledger_update = _make_ledger_update(
            agent="human_supervisor",
            action="directive",
            content=human_input,
            target=directive.specialist or directive.action,
        )

        # If the human asks to generate the report, go directly
        if directive.action == "generate_report":
            return Command(
                goto="final_report_generation",
                update={
                    "messages": [AIMessage(content="Generating final report as requested.")],
                    "human_supervisor_messages": [
                        HumanMessage(content=human_input),
                        AIMessage(content="Proceeding to final report generation."),
                    ],
                    **ledger_update,
                },
            )

        # Otherwise route to directive processing
        return Command(
            goto="process_human_directive",
            update={
                "human_supervisor_messages": [
                    HumanMessage(content=human_input),
                    AIMessage(
                        content=f"Processing your directive: {directive.action} — {directive.content}"
                    ),
                ],
                # Store the directive details in messages for the processor
                "messages": [
                    HumanMessage(
                        content=(
                            f"[Human Supervisor Directive]\n"
                            f"Action: {directive.action}\n"
                            f"Content: {directive.content}\n"
                            f"Specialist: {directive.specialist or 'N/A'}"
                        )
                    ),
                ],
                **ledger_update,
            },
        )

    # Empty input — generate report by default
    ledger_update = _make_ledger_update(
        agent="human_supervisor",
        action="generate_report",
        content="No further instructions received. Generating final report.",
    )
    return Command(
        goto="final_report_generation",
        update={
            "messages": [AIMessage(content="No further instructions received. Generating final report.")],
            **ledger_update,
        },
    )


async def process_human_directive(
    state: AgentState, config: RunnableConfig,
) -> Command[Literal["human_supervisor"]]:
    """Execute the action requested by the human supervisor.

    Parses the most recent human directive from state messages and executes
    the corresponding action (research, specialist query, negotiation, etc.).
    Results are added to state and control returns to the human supervisor.

    Args:
        state: Current agent state containing the human directive
        config: Runtime configuration

    Returns:
        Command to loop back to human_supervisor with results
    """
    # Extract the most recent directive from messages
    messages = state.get("messages", [])
    directive_message = None
    for msg in reversed(messages):
        if hasattr(msg, "content") and "[Human Supervisor Directive]" in str(msg.content):
            directive_message = str(msg.content)
            break

    if not directive_message:
        return Command(
            goto="human_supervisor",
            update={
                "messages": [AIMessage(content="No directive found. Please provide an instruction.")],
            },
        )

    # Parse the directive
    lines = directive_message.split("\n")
    action = ""
    content = ""
    specialist = None
    for line in lines:
        if line.startswith("Action: "):
            action = line[len("Action: "):]
        elif line.startswith("Content: "):
            content = line[len("Content: "):]
        elif line.startswith("Specialist: "):
            spec = line[len("Specialist: "):]
            if spec != "N/A":
                specialist = spec

    update: Dict[str, Any] = {}
    result_message = ""

    # Build a SupervisorState-like dict for reusing existing helper functions
    supervisor_state: SupervisorState = {
        "supervisor_messages": state.get("supervisor_messages", []),
        "research_brief": state.get("research_brief", ""),
        "notes": state.get("notes", []),
        "raw_notes": state.get("raw_notes", []),
        "research_iterations": 0,
        "specialist_queries": [],
        "specialist_responses": [],
        "negotiation_round": state.get("negotiation_round", 0),
        "negotiation_max_rounds": state.get("negotiation_max_rounds", 2),
        "negotiation_messages": state.get("negotiation_messages", []),
        "geneticist_proposals": state.get("geneticist_proposals", []),
        "systems_theorist_proposals": state.get("systems_theorist_proposals", []),
        "predictive_cognition_proposals": state.get("predictive_cognition_proposals", []),
        "critiques": state.get("critiques", []),
        "hypotheses_bundle": state.get("hypotheses_bundle"),
    }

    if action == "conduct_research":
        # Delegate research to a sub-researcher
        try:
            tool_result = await researcher_subgraph.ainvoke(
                {
                    "researcher_messages": [HumanMessage(content=content)],
                    "research_topic": content,
                },
                config,
            )
            compressed = tool_result.get(
                "compressed_research",
                "No results returned from research.",
            )
            raw = tool_result.get("raw_notes", [])
            result_message = f"**Research Results:**\n{compressed}"
            update["notes"] = [compressed]
            if raw:
                update["raw_notes"] = raw
        except Exception as e:
            result_message = f"Error conducting research on '{content[:100]}': {e}"

    elif action == "query_specialist":
        if not specialist:
            result_message = (
                "Please specify which specialist to query "
                "(geneticist, systems_theorist, or predictive_cognition)."
            )
        else:
            try:
                response = await handle_specialist_query(
                    specialist_role=specialist,
                    question=content,
                    state=supervisor_state,
                    config=config,
                )
                specialist_name = specialist.replace("_", " ").title()
                result_message = f"**{specialist_name} Response:**\n{response}"
            except Exception as e:
                result_message = f"Error querying {specialist} with question '{content[:100]}': {e}"

    elif action == "conduct_negotiation_round":
        try:
            round_result = await conduct_single_negotiation_round(
                state=supervisor_state,
                config=config,
                round_instructions=content,
            )
            current_round = round_result.get(
                "negotiation_round",
                state.get("negotiation_round", 0),
            )
            num_messages = len(round_result.get("negotiation_messages", []))
            result_message = (
                f"**Negotiation Round {current_round} Complete**\n"
                f"Instructions: {content}\n"
                f"Generated {num_messages} specialist contributions.\n\n"
            )
            for msg in round_result.get("negotiation_messages", [])[:MAX_PREVIEW_MESSAGES]:
                if hasattr(msg, "content"):
                    msg_content = str(msg.content)[:300]
                    result_message += f"{msg_content}\n\n"

            for key in [
                "negotiation_round", "negotiation_messages",
                "geneticist_proposals", "systems_theorist_proposals",
                "predictive_cognition_proposals", "critiques",
            ]:
                if key in round_result:
                    update[key] = round_result[key]
        except Exception as e:
            result_message = f"Error conducting negotiation round with instructions '{content[:100]}': {e}"

    elif action == "recall_from_negotiation":
        try:
            recall_result = await recall_from_negotiation(
                state=supervisor_state,
                config=config,
                query=content,
                specialist_filter=specialist,
            )
            filter_text = f" (filtered to {specialist})" if specialist else ""
            result_message = (
                f"**Recall Result{filter_text}:**\n"
                f"Query: {content}\n\n{recall_result}"
            )
        except Exception as e:
            result_message = f"Error recalling with query '{content[:100]}' (specialist: {specialist or 'all'}): {e}"

    elif action == "synthesize_negotiation":
        try:
            hypotheses_bundle = await synthesize_negotiation(
                state=supervisor_state,
                config=config,
                synthesis_instructions=content,
            )
            num_h = len(hypotheses_bundle.hypotheses)
            num_p = len(hypotheses_bundle.predictions)
            num_d = len(hypotheses_bundle.disagreements)
            result_message = (
                f"**Synthesis Complete**\n"
                f"Generated {num_h} hypotheses, {num_p} predictions, "
                f"{num_d} documented disagreements.\n"
            )
            if hypotheses_bundle.hypotheses:
                result_message += "Hypotheses: " + ", ".join(
                    h.id for h in hypotheses_bundle.hypotheses
                ) + "\n"
            update["hypotheses_bundle"] = hypotheses_bundle
        except Exception as e:
            result_message = f"Error synthesizing with instructions '{content[:100]}': {e}"

    elif action == "provide_feedback":
        # Feedback is stored in messages for context
        result_message = (
            f"**Feedback noted.** Your guidance has been recorded:\n{content}"
        )

    else:
        result_message = (
            f"Unknown action '{action}'. Please provide a valid instruction."
        )

    update["messages"] = [AIMessage(content=result_message)]
    update["human_supervisor_messages"] = [AIMessage(content=result_message)]

    # Log the result to the forum ledger
    ledger_update = _make_ledger_update(
        agent=action if action == "query_specialist" and specialist else "research_supervisor",
        action=f"directive_result:{action}",
        content=result_message,
        target="human_supervisor",
        metadata={"directive_action": action},
    )
    update.update(ledger_update)

    return Command(goto="human_supervisor", update=update)


# Main Deep Researcher Graph Construction
# ──────────────────────────────────────────────────────────────────────
# Forum-based topology: every agent node is reachable from the human
# supervisor via conditional, interruptible edges.  The graph is a
# full (nondirected, frequently cyclical) graph — not a chain.
#
#          ┌─────────────────────────────────────────────────┐
#          │              HUMAN SUPERVISOR                   │
#          │  (central hub — edge to every other node)       │
#          └──┬──────┬──────────┬─────────────┬─────────────┘
#             │      │          │             │
#             ▼      ▼          ▼             ▼
#       research   process   final_report  clarify
#       supervisor  human    generation    _with_user
#             ▲    directive    ▲
#             │      │          │
#             └──────┘          │
#                    └──────────┘
# ──────────────────────────────────────────────────────────────────────

deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes — each represents an *agent* (not a subtask)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # Clarification agent
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Brief-writing agent
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research supervisor agent
deep_researcher_builder.add_node("human_supervisor", human_supervisor)             # Human supervisor agent (central hub)
deep_researcher_builder.add_node("process_human_directive", process_human_directive)  # Directive-processing agent
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report-generation agent

# Define edges — conditional edges are handled via Command routing in
# the node functions.  The human_supervisor node has edges to every
# other node (research_supervisor, process_human_directive,
# final_report_generation) through its Command returns.
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "human_supervisor")        # After research, human decides next step
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher forum
deep_researcher = deep_researcher_builder.compile()