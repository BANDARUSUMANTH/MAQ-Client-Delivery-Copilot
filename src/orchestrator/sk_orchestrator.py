import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
# Legacy compatibility patch for scipy/flaml in newer NumPy versions
for attr in ["long", "ulong"]:
    if not hasattr(np, attr):
        setattr(np, attr, int)

import os
import re
import json
from jinja2 import Environment, FileSystemLoader
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable

from src.config import get_llm_config, ROOT_DIR
from src.ingestion.ingest import perform_hybrid_rag
from src.agents.retrieval_agent import run_retrieval_agent
from src.telemetry import logger, log_query, log_risk_detected

def parse_retrieved_projects(rag_nodes):
    """
    Groups and extracts structured project metadata and timesheet details
    from the retrieved Hybrid RAG nodes.
    """
    projects_dict = {}
    for node in rag_nodes:
        meta = node.get("metadata", {})
        proj_id = meta.get("project_id")
        if not proj_id:
            continue
            
        if proj_id not in projects_dict:
            projects_dict[proj_id] = {
                "id": proj_id,
                "name": meta.get("project_name", "Unknown Project"),
                "manager": "Unknown",
                "client": "Unknown",
                "health": "Green",
                "start_date": "N/A",
                "end_date": "N/A",
                "budget_hours": 0,
                "scope": "",
                "timesheets": [],
                "logged_hours": 0,
                "timesheet_summary": "No timesheet records retrieved."
            }
            
        # Parse fields from the document text
        text = node.get("text", "")
        
        if meta.get("type") == "sharepoint_project":
            # Extract fields
            mgr_match = re.search(r"Manager:\s*(.*)", text)
            client_match = re.search(r"Client:\s*(.*)", text)
            health_match = re.search(r"Health Status:\s*(.*)", text)
            timeline_match = re.search(r"Timeline:\s*(.*)\s*to\s*(.*)", text)
            budget_match = re.search(r"Budgeted Hours:\s*(\d+)", text)
            scope_match = re.search(r"Scope:\s*(.*)", text, re.DOTALL)
            
            if mgr_match: projects_dict[proj_id]["manager"] = mgr_match.group(1).strip()
            if client_match: projects_dict[proj_id]["client"] = client_match.group(1).strip()
            if health_match: projects_dict[proj_id]["health"] = health_match.group(1).strip()
            if timeline_match:
                projects_dict[proj_id]["start_date"] = timeline_match.group(1).strip()
                projects_dict[proj_id]["end_date"] = timeline_match.group(2).strip()
            if budget_match: projects_dict[proj_id]["budget_hours"] = int(budget_match.group(1))
            if scope_match: projects_dict[proj_id]["scope"] = scope_match.group(1).strip()
            
        elif meta.get("type") == "d365_timesheet":
            # Extract timesheet specific records
            hours_match = re.search(r"Hours Logged:\s*(\d+)", text)
            consultant_match = re.search(r"Consultant:\s*(.*)", text)
            task_match = re.search(r"Task:\s*(.*)", text)
            
            hours = int(hours_match.group(1)) if hours_match else 0
            consultant = consultant_match.group(1).strip() if consultant_match else "Unknown"
            task = task_match.group(1).strip() if task_match else "Consulting task"
            
            projects_dict[proj_id]["timesheets"].append({
                "consultant": consultant,
                "hours": hours,
                "task": task
            })
            projects_dict[proj_id]["logged_hours"] += hours

    # Format timesheet summary text
    for proj_id, proj in projects_dict.items():
        ts_entries = proj["timesheets"]
        if ts_entries:
            summary = f"Total of {len(ts_entries)} entries logged. "
            # Group by consultant
            consultants_hours = {}
            for ts in ts_entries:
                c = ts["consultant"]
                consultants_hours[c] = consultants_hours.get(c, 0) + ts["hours"]
            
            cons_str = ", ".join([f"{c} ({h} hrs)" for c, h in consultants_hours.items()])
            summary += f"Consultants active: {cons_str}."
            proj["timesheet_summary"] = summary
            
    return list(projects_dict.values())

def parse_devops_text(devops_summary):
    """
    Parses structural values out of DevOps retrieval text summary 
    to feed into HTML visualizations.
    """
    planned = 0
    completed = 0
    sprint_name = "Sprint Info"
    work_items = []
    
    # Parse Sprint Name
    sprint_match = re.search(r"Sprint:\s*(.*)", devops_summary)
    if sprint_match:
        sprint_name = sprint_match.group(1).strip()
        
    # Parse Story Points
    planned_match = re.search(r"Planned Points:\s*([\d\.]+)", devops_summary)
    completed_match = re.search(r"Completed Points:\s*([\d\.]+)", devops_summary)
    
    if planned_match: planned = int(float(planned_match.group(1)))
    if completed_match: completed = int(float(completed_match.group(1)))
    
    # Parse individual work items
    # Example format: - [Done] WI #101: Configure Azure SQL connection | Assigned: Alice Miller | Points: 5
    wi_pattern = r"-\s*\[(.*?)\]\s*WI\s*#(\d+):\s*(.*?)\s*\|\s*Assigned:\s*(.*?)\s*\|\s*Points:\s*([\d\.]+)"
    matches = re.finditer(wi_pattern, devops_summary)
    for m in matches:
        work_items.append({
            "status": m.group(1).strip(),
            "id": int(m.group(2)),
            "title": m.group(3).strip(),
            "assigned_to": m.group(4).strip(),
            "points": int(float(m.group(5)))
        })
        
    return sprint_name, planned, completed, work_items

def run_rule_based_synthesis(project, devops_summary):
    """Fallback analytical warning generator if LLM key is absent."""
    health_color = project["health"].lower()
    
    # Calculate ratios
    logged = project["logged_hours"]
    budget = project["budget_hours"]
    hours_pct = (logged / budget * 100) if budget > 0 else 0
    
    sprint_name, planned, completed, work_items = parse_devops_text(devops_summary)
    if planned > 0:
        sprint_pct = (completed / planned * 100)
    elif len(work_items) > 0:
        completed_count = sum(1 for wi in work_items if wi["status"].lower() in ["closed", "done", "completed", "resolved"])
        sprint_pct = (completed_count / len(work_items) * 100)
        planned = len(work_items)
        completed = completed_count
    else:
        sprint_pct = 0
    
    warnings = []
    
    # Check overwork (D365 timesheets)
    extreme_ot = False
    for ts in project["timesheets"]:
        if ts["hours"] > 50:
            extreme_ot = True
            warnings.append(f"Consultant {ts['consultant']} is logging excessive hours ({ts['hours']} hrs) in timesheets, indicating burn-out risk.")
            
    # Check budget burn
    if hours_pct > 80:
        warnings.append(f"Project budget is heavily depleted: {logged} hours logged out of {budget} budgeted ({hours_pct:.1f}% burn rate).")
        
    # Check sprint slippage
    blocked_count = sum(1 for wi in work_items if wi["status"] == "Blocked")
    if blocked_count > 0:
        warnings.append(f"There are {blocked_count} blocked work items in {sprint_name}.")
        
    if sprint_pct < 50 and planned > 0:
        warnings.append(f"Sprint completion velocity is dangerously low at {sprint_pct:.1f}%.")

    # Determine final color
    risk_level = "Green"
    if health_color == "red" or len(warnings) >= 3 or (sprint_pct < 30 and planned > 0):
        risk_level = "Red"
    elif health_color == "amber" or len(warnings) > 0:
        risk_level = "Amber"
        
    # Synthesize text
    if not warnings:
        synthesis = "Overall project delivery is tracking normally. Resource workloads are balanced, and sprint velocity aligns with planning."
    else:
        synthesis = f"Alert: Risk factors identified! " + " ".join(warnings)
        
    # Log risk to telemetry
    if risk_level in ["Red", "Amber"]:
        log_risk_detected(project["name"], risk_level, synthesis)
        
    return risk_level, synthesis

def generate_mitigation_plan(project, risk_level, devops_summary):
    plans = []
    
    # Check timesheet overtime (D365)
    for ts in project.get("timesheets", []):
        if ts["hours"] > 50:
            plans.append(f"Resource Burn-out Risk: {ts['consultant']} is working {ts['hours']} hours/week. Suggest assigning a secondary developer to share workload.")
            
    # Check budget burn
    logged = project.get("logged_hours", 0)
    budget = project.get("budget_hours", 0)
    hours_pct = (logged / budget * 100) if budget > 0 else 0
    if hours_pct > 80:
        plans.append(f"Budget Overrun Warning: {hours_pct:.1f}% of hours are consumed. Schedule a scope containment review with the client lead.")
        
    # Check DevOps sprint completion & slippage
    sprint_name, planned, completed, work_items = parse_devops_text(devops_summary)
    sprint_pct = (completed / planned * 100) if planned > 0 else 0
    
    blocked_items = [wi for wi in work_items if wi["status"] == "Blocked"]
    if blocked_items:
        plans.append(f"Blocked Sprint Tasks: {len(blocked_items)} items blocked in {sprint_name}. Escalate blockages internally to unblock team.")
        
    if sprint_pct < 50 and planned > 0:
        plans.append(f"Low Sprint Velocity: Completion rate is {sprint_pct:.1f}%. Re-evaluate sprint capacity and deprioritize non-essential items.")
        
    if not plans:
        plans.append("Project is running smoothly. Continue tracking weekly metrics.")
        
    return plans

async def run_semantic_kernel_synthesis(kernel, project, devops_summary):
    """Uses Semantic Kernel to run an LLM prompt and synthesize signals into a risk summary."""
    prompt = """
    You are an expert project risk auditor at MAQ Software. 
    Analyze the following project delivery signals to detect underlying risks:
    
    --- PROJECT DETAILS ---
    Project: {{ $project_name }} (Manager: {{ $manager }})
    SharePoint Health: {{ $sp_health }}
    Budget Hours: {{ $budget_hours }}
    Logged Hours: {{ $logged_hours }}
    Timesheet Summary: {{ $timesheet_summary }}
    
    --- DEVOPS STATUS ---
    {{ $devops_summary }}
    
    Provide a concise risk synthesis (3-4 sentences). 
    Evaluate whether timesheet logging rates match work item completions.
    Highlight resource over-allocation (e.g. anyone logging >50 hours/week) or project delays.
    Suggest a risk classification (Red, Amber, or Green). Output ONLY the synthesis paragraph.
    """
    
    # Define prompt template
    prompt_config = PromptTemplateConfig(
        template=prompt,
        input_variables=[
            InputVariable(name="project_name", description="Project Name", is_required=True),
            InputVariable(name="manager", description="Project Manager", is_required=True),
            InputVariable(name="sp_health", description="Sharepoint Health", is_required=True),
            InputVariable(name="budget_hours", description="Budget Hours", is_required=True),
            InputVariable(name="logged_hours", description="Logged Hours", is_required=True),
            InputVariable(name="timesheet_summary", description="Timesheet log summaries", is_required=True),
            InputVariable(name="devops_summary", description="DevOps work items and sprints", is_required=True),
        ]
    )
    
    # Create the function
    synthesis_function = kernel.add_function(
        function_name="synthesize_risk",
        plugin_name="RiskAuditingPlugin",
        prompt_template_config=prompt_config
    )
    
    # Invoke
    result = await kernel.invoke(
        synthesis_function,
        project_name=project["name"],
        manager=project["manager"],
        sp_health=project["health"],
        budget_hours=str(project["budget_hours"]),
        logged_hours=str(project["logged_hours"]),
        timesheet_summary=project["timesheet_summary"],
        devops_summary=devops_summary
    )
    
    synthesis_text = str(result).strip()
    
    # Classify color based on LLM output words
    synthesis_lower = synthesis_text.lower()
    if "red" in synthesis_lower or "severe risk" in synthesis_lower:
        risk_color = "Red"
    elif "amber" in synthesis_lower or "warning" in synthesis_lower or "moderate risk" in synthesis_lower:
        risk_color = "Amber"
    else:
        risk_color = "Green"
        
    if risk_color in ["Red", "Amber"]:
        log_risk_detected(project["name"], risk_color, synthesis_text)
        
    return risk_color, synthesis_text

async def generate_status_report(query: str, user_id: str, return_data: bool = False):
    """
    Main Orchestrator Entrypoint:
    1. Log query.
    2. Perform Hybrid RAG.
    3. Delegate to AutoGen Retrieval Agent.
    4. Run SK or fallback rule engine for analysis.
    5. Compile HTML dashboard report.
    """
    log_query(user_id, query)
    
    # 1. Hybrid RAG (filters by user_id)
    retrieved_nodes = perform_hybrid_rag(query, user_id, top_k=10)
    if not retrieved_nodes:
        logger.warning(f"No records returned for query: '{query}' under user: '{user_id}'")
        return None
        
    # 2. Group records into structured projects
    projects = parse_retrieved_projects(retrieved_nodes)
    
    # 3. Setup Semantic Kernel if LLM is active
    sk_kernel = None
    try:
        cfg = get_llm_config()
        sk_kernel = Kernel()
        if cfg["provider"] == "azure_openai":
            sk_kernel.add_service(
                AzureChatCompletion(
                    service_id="default",
                    deployment_name=cfg["deployment"],
                    endpoint=cfg["endpoint"],
                    api_key=cfg["api_key"],
                    api_version=cfg["api_version"]
                )
            )
            logger.info("Semantic Kernel initialized with Azure OpenAI Service.")
        elif cfg["provider"] == "openai":
            sk_kernel.add_service(
                OpenAIChatCompletion(
                    service_id="default",
                    ai_model_id=cfg["model"],
                    api_key=cfg["api_key"]
                )
            )
            logger.info("Semantic Kernel initialized with OpenAI Service.")
        elif cfg["provider"] == "ollama":
            import openai
            client = openai.AsyncOpenAI(
                base_url=cfg["endpoint"],
                api_key="ollama"
            )
            sk_kernel.add_service(
                OpenAIChatCompletion(
                    service_id="default",
                    ai_model_id=cfg["model"],
                    async_client=client
                )
            )
            logger.info("Semantic Kernel initialized with Local Ollama Service.")
    except Exception as e:
        logger.info(f"Skipping Semantic Kernel LLM setup (running local analysis): {e}")
        sk_kernel = None

    # 4. Synthesize signals for each project
    enriched_projects = []
    for proj in projects:
        proj_id = proj["id"]
        
        # Call AutoGen Retrieval Agent for DevOps sprint summaries
        devops_summary = run_retrieval_agent(
            query=f"Get sprint status details for {proj['name']}",
            project_id=proj_id,
            user_id=user_id
        )
        
        # Analyze and synthesize risk (SK or Rule-based)
        if sk_kernel:
            try:
                risk_level, synthesis = await run_semantic_kernel_synthesis(sk_kernel, proj, devops_summary)
            except Exception as ex:
                logger.error(f"Semantic Kernel synthesis failed: {ex}. Falling back to rules.")
                risk_level, synthesis = run_rule_based_synthesis(proj, devops_summary)
        else:
            risk_level, synthesis = run_rule_based_synthesis(proj, devops_summary)
            
        # Parse devops fields for HTML visualization
        sprint_name, planned_pts, completed_pts, work_items = parse_devops_text(devops_summary)
        
        # Calculate visualization variables
        budget = proj["budget_hours"]
        logged = proj["logged_hours"]
        hours_pct = int((logged / budget * 100)) if budget > 0 else 0
        hours_pct_capped = min(hours_pct, 100)
        
        # If story points are not configured (0), fall back to checking the count of work items
        if planned_pts > 0:
            sprint_pct = int((completed_pts / planned_pts * 100))
        elif len(work_items) > 0:
            completed_count = sum(1 for wi in work_items if wi["status"].lower() in ["closed", "done", "completed", "resolved"])
            sprint_pct = int((completed_count / len(work_items) * 100))
            planned_pts = len(work_items)
            completed_pts = completed_count
        else:
            sprint_pct = 0
            
        sprint_pct_capped = min(sprint_pct, 100)
        
        # Determine visual CSS bar colors
        hours_bar_color = "green"
        if hours_pct > 100: hours_bar_color = "red"
        elif hours_pct > 80: hours_bar_color = "amber"
        
        sprint_bar_color = "green"
        if sprint_pct < 40: sprint_bar_color = "red"
        elif sprint_pct < 70: sprint_bar_color = "amber"
        
        enriched_projects.append({
            "name": proj["name"],
            "client": proj["client"],
            "manager": proj["manager"],
            "health": risk_level,
            "health_color": risk_level.lower(),
            "start_date": proj["start_date"],
            "end_date": proj["end_date"],
            "budget_hours": budget,
            "logged_hours": logged,
            "hours_percentage": hours_pct,
            "hours_percentage_capped": hours_pct_capped,
            "hours_bar_color": hours_bar_color,
            "timesheet_summary": proj["timesheet_summary"],
            
            "sprint_name": sprint_name,
            "planned_points": planned_pts,
            "completed_points": completed_pts,
            "sprint_percentage": sprint_pct,
            "sprint_percentage_capped": sprint_pct_capped,
            "sprint_bar_color": sprint_bar_color,
            "work_items": work_items,
            "risk_synthesis": synthesis,
            "mitigation_plan": generate_mitigation_plan(proj, risk_level, devops_summary)
        })
        
    if return_data:
        return enriched_projects

    # 5. Render HTML template using Jinja2
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")
    
    html_output = template.render(
        user_id=user_id,
        projects=enriched_projects
    )
    
    # Save output to workspace
    output_path = ROOT_DIR / "delivery_status_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)
        
    logger.info(f"Stunning HTML status report generated at: {output_path}")
    return output_path

if __name__ == "__main__":
    import asyncio
    # Simple direct test run
    logger.info("Executing main orchestrator test...")
    
    # Trigger ingestion to make sure DB is ready
    from src.ingestion.ingest import ingest_data
    ingest_data()
    
    # Manager asks: "What is the health of active Power BI delivery projects?"
    report_file = asyncio.run(generate_status_report(
        query="What is the health of active Power BI delivery projects?",
        user_id="john_doe" # john_doe is authorized for Power BI Dashboard (P001)
    ))
    print(f"Report file: {report_file}")
