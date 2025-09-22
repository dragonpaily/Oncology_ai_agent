# In src/app.py

import gradio as gr

import os
from langchain_google_genai import ChatGoogleGenerativeAI # <-- IMPORT THE CORRECT LIBRARY
from langchain.agents import AgentExecutor, create_tool_calling_agent # <-- IMPORT AGENT HELPERS
from langchain_core.prompts import ChatPromptTemplate # <-- IMPORT PROMPT HELPER
from dotenv import load_dotenv
# --- 1. Import your unified toolbox ---
from .agent_tools import all_tools
load_dotenv() #loads variables from the .env folder
try:
    # The ChatGoogleGenerativeAI class will automatically find and use this key,
    # so we just need to check if it was loaded successfully.
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
    print("✅ Gemini API key found in environment.")
except Exception as e:
    print(f"❗️ ERROR: Could not configure Gemini API. {e}")

# --- 3. Update the Agent's "Brain" to know about the new tool ---
SYSTEM_PROMPT_V4 = """
You are a world-class oncology AI assistant. Your purpose is to act as a single, unified system to analyze multimodal data.

You have a comprehensive suite of tools:
- `generate_qualitative_report_tool`: Use this for a high-level visual summary of MRI scans.
- `run_segmentation_analysis`: Use this for precise, quantitative tumor volume and location data from MRIs.
- `calculate_percentage_change`: Use to calculate tumor progression after getting volumes.
- `clinical_guideline_retriever_tool`: Use to search your internal knowledge base.
- `pubmed_search_tool`: Use to search for the latest research.
- `oncokb_query_tool`: Use to find therapies for known genetic mutations.
- `histopathology_mutation_analyzer_tool`: Use to predict mutations from histology slides.

**Your Process:**
Analyze the user's query and context, formulate a plan, execute the plan using your tools in a logical sequence, and synthesize all findings into a single, comprehensive report. You MUST NOT give a final diagnosis.
"""

# The system prompt is correct, but we'll format it for the new agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_V4),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Initialize the LLM using the LangChain-compatible class
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")


# Create the agent
agent = create_tool_calling_agent(llm, all_tools, prompt)

# Create the Agent Executor, which runs the agent
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)



# --- 5. The UPDATED Main Logic Function ---
def run_agent_flow(clinical_summary, user_query, files_dict):
    """
    This function takes all the inputs from Gradio and runs the agent executor.
    """
    # Construct the detailed input for the agent
    full_input = f"""
    CONTEXT:
    Clinical Summary: "{clinical_summary}"
    File Paths: { {k: v.name for k, v in files_dict.items()} }

    USER QUERY:
    "{user_query}"

    Please process this request by forming a plan and using your available tools to generate a single, comprehensive report.
    """

    # Invoke the agent executor with the input
    response = agent_executor.invoke({"input": full_input})

    # The final answer is in the 'output' key
    return response['output']

# In src/app.py

# --- 6. The Complete Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Unified AI Agent for Clinical Decision Support")
    gr.Markdown("This single, powerful agent uses a suite of tools to generate one comprehensive report from multimodal data.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Patient Data")
            summary_input = gr.Textbox(label="Clinical Summary", lines=5, value="A 55-year-old male with a history of headaches. An initial scan revealed a lesion. A follow-up scan was performed 6 months later.")
            query_input = gr.Textbox(label="Your Query for the Agent", value="Provide a full progression report by analyzing the initial and follow-up scans. Check the knowledge base for treatment options based on the tumor's volume change.")
            
            # --- CORRECTED: ALL FOUR SCAN TYPES NOW INCLUDED ---
            gr.Markdown("**Initial Scans**")
            t1c1_input, t1n1_input, t2f1_input, t2w1_input = [gr.File(label="T1c"), gr.File(label="T1n"), gr.File(label="FLAIR"), gr.File(label="T2w")]
            
            gr.Markdown("**Follow-up Scans**")
            t1c3_input, t1n3_input, t2f3_input, t2w3_input = [gr.File(label="T1c"), gr.File(label="T1n"), gr.File(label="FLAIR"), gr.File(label="T2w")]
            
            gr.Markdown("**Histopathology (Optional)**")
            histo1_input = gr.File(label="Histology Slide")
            
            btn = gr.Button("Run Unified Agent", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Unified Agent Report")
            output_report = gr.Textbox(label="Agent's Comprehensive Report", lines=25, interactive=False)

    # --- CORRECTED: HANDLER ACCEPTS ALL 9 FILE INPUTS ---
    def gradio_interface_handler(summary, query, t1c1, t1n1, t2f1, t2w1, t1c3, t1n3, t2f3, t2w3, histo1):
        files = {
            't1c1': t1c1, 't1n1': t1n1, 't2f1': t2f1, 't2w1': t2w1,
            't1c3': t1c3, 't1n3': t1n3, 't2f3': t2f3, 't2w3': t2w3,
            'histo1': histo1
        }
        uploaded_files = {k: v for k, v in files.items() if v is not None}
        return run_agent_flow(summary, query, uploaded_files)

    # --- CORRECTED: CLICK EVENT PASSES ALL 11 INPUTS ---
    btn.click(
        gradio_interface_handler,
        inputs=[
            summary_input, query_input, 
            t1c1_input, t1n1_input, t2f1_input, t2w1_input,
            t1c3_input, t1n3_input, t2f3_input, t2w3_input,
            histo1_input
        ],
        outputs=[output_report]
    )

# --- 7. Launch the App ---
if __name__ == "__main__":
    demo.launch(debug=True)