import numpy as np
# Legacy compatibility patch for scipy/flaml in newer NumPy versions
for attr in ["long", "ulong"]:
    if not hasattr(np, attr):
        setattr(np, attr, int)

import os
import autogen
import pandas as pd
from src.config import get_llm_config, DATA_DIR
from src.mcp.devops_server import get_sprint_data
from src.telemetry import logger

def build_autogen_config():
    """Builds the llm_config dictionary for AutoGen based on selected provider."""
    try:
        cfg = get_llm_config()
    except ValueError as e:
        logger.warning(f"AutoGen LLM config warning: {e}. Falling back to mock/no-op LLM for offline testing.")
        return None

    config_list = []
    if cfg["provider"] == "azure_openai":
        config_list = [{
            "model": cfg["deployment"],
            "api_key": cfg["api_key"],
            "base_url": cfg["endpoint"],
            "api_type": "azure",
            "api_version": cfg["api_version"]
        }]
    elif cfg["provider"] == "openai":
        config_list = [{
            "model": cfg["model"],
            "api_key": cfg["api_key"]
        }]
    elif cfg["provider"] == "gemini":
        config_list = [{
            "model": cfg["model"],
            "api_key": cfg["api_key"],
            "api_type": "google"
        }]
    elif cfg["provider"] == "ollama":
        config_list = [{
            "model": cfg["model"],
            "base_url": cfg["endpoint"],
            "api_key": "ollama"
        }]

    return {
        "config_list": config_list,
        "temperature": 0.0,
        "timeout": 60
    }

def check_user_authorization(project_id: str, user_id: str) -> bool:
    """Verifies if user_id is authorized to view project_id details."""
    sharepoint_path = DATA_DIR / "sharepoint_projects.csv"
    if not sharepoint_path.exists():
        return False
    
    try:
        df = pd.read_csv(sharepoint_path)
        # Match project_id (e.g. P001) or ProjectName
        project_row = df[(df["ProjectId"] == project_id) | (df["ProjectName"] == project_id)]
        if project_row.empty:
            return False
            
        auth_users_str = project_row.iloc[0]["AuthorizedUsers"]
        authorized_users = [u.strip() for u in str(auth_users_str).split(",")]
        return user_id in authorized_users
    except Exception as e:
        logger.error(f"Error checking authorization in AutoGen: {e}")
        return False

def run_retrieval_agent(query: str, project_id: str, user_id: str) -> str:
    """
    Spawns the AutoGen Data Retrieval Agent, performs authorization checks,
    executes the DevOps FastMCP tool if authorized, and returns summarized answers.
    """
    logger.info(f"Invoking AutoGen Retrieval Agent for project {project_id} and user {user_id}")
    
    # 1. Enforce auth check before doing anything
    if not check_user_authorization(project_id, user_id):
        logger.warning(f"Security Alert: User {user_id} denied access to DevOps data for {project_id}")
        return f"ACCESS DENIED: User '{user_id}' does not have permissions to access DevOps data for project '{project_id}'."

    # 2. Get LLM config
    llm_config = build_autogen_config()
    
    # Fallback/Mock behavior if no real LLM config is loaded (to avoid crash during initial local demo setup)
    if not llm_config:
        logger.info("Executing rule-based retrieval agent (No LLM API keys configured)")
        # Direct execution of the DevOps tool if authorized
        devops_data = get_sprint_data(project_id)
        return (
            f"[AutoGen Retrieval Agent Response (Rule-based Fallback)]\n"
            f"Authorized access confirmed for user '{user_id}' on project '{project_id}'.\n"
            f"DevOps data retrieved: \n{devops_data}"
        )

    # 3. Define AutoGen Agents
    assistant = autogen.AssistantAgent(
        name="data_retrieval_agent",
        llm_config=llm_config,
        system_message=(
            "You are the Data Retrieval Agent for MAQ Software.\n"
            "Your job is to fetch and summarize Azure DevOps sprint and task data for a project.\n"
            "You must use the get_sprint_data tool to fetch the latest sprint and work items.\n"
            "Provide a concise summary of the sprint status, planned vs completed story points, "
            "and list any blocked or in-progress tasks. Terminate with TERMINATE when done."
        )
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "") or not x.get("content"),
        code_execution_config=False
    )

    # 4. Register the MCP Tool function on agents
    # Define a wrapper that carries project_id to avoid agent hallucinations of arguments
    def retrieve_tool_wrapper():
        return get_sprint_data(project_id)

    autogen.agentchat.register_function(
        retrieve_tool_wrapper,
        caller=assistant,
        executor=user_proxy,
        name="fetch_sprint_data",
        description="Fetches sprint metrics for the currently requested project."
    )

    # 5. Execute chat with robust fallback
    try:
        user_proxy.initiate_chat(
            assistant,
            message=chat_prompt
        )
        
        # 6. Retrieve the assistant's final response
        chat_history = user_proxy.chat_messages[assistant]
        for msg in reversed(chat_history):
            content = msg.get("content", "")
            if content and "fetch_sprint_data" not in content and "data_retrieval_agent" not in msg.get("name", "").lower():
                cleaned_content = content.replace("TERMINATE", "").strip()
                if cleaned_content and len(cleaned_content) > 30:
                    return cleaned_content
    except Exception as chat_ex:
        logger.warning(f"AutoGen retrieval chat encountered an error: {chat_ex}. Invoking direct fallback.")

    # Direct fallback execution
    logger.info("Executing direct fallback tool retrieval (bypass LLM function-calling).")
    devops_data = get_sprint_data(project_id)
    return (
        f"[AutoGen Fallback Summary]\n"
        f"Sprint: Active Sprint\n"
        f"DevOps data retrieved directly:\n{devops_data}"
    )

if __name__ == "__main__":
    # Local quick test
    print("Testing AutoGen Retrieval Agent...")
    res = run_retrieval_agent(
        query="Fetch sprint status for Sales Dashboard",
        project_id="P001",
        user_id="john_doe"
    )
    print("\nResult:")
    print(res)
