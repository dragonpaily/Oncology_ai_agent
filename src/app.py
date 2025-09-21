# In src/app.py

import gradio as gr
import google.generativeai as genai
import os

# --- 1. Import your unified toolbox ---
from agent_tools import all_tools

# --- 2. Handle API Key Locally (same as before) ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY", "api key here")
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured.")
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

# --- 4. Initialize the UNIFIED Agent Model ---
model_agent = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    tools=all_tools,
    system_instruction=SYSTEM_PROMPT_V4
)

# --- 5. The Simplified Main Logic Function ---
def run_unified_agent_flow(clinical_summary, user_query, files_dict):
    chat = model_agent.start_chat(enable_automatic_function_calling=True)
    
    # Construct a single, detailed prompt
    prompt = f"""
    CONTEXT:
    Clinical Summary: "{clinical_summary}"
    File Paths: { {k: v.name for k, v in files_dict.items()} }

    USER QUERY:
    "{user_query}"

    Please process this request by forming a plan and using your available tools to generate a single, comprehensive report.
    """
    
    response = chat.send_message(prompt)
    return response.text

# --- 6. The Simplified Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Unified AI Agent for Clinical Decision Support")
    gr.Markdown("This single, powerful agent uses a suite of tools to generate one comprehensive report from multimodal data.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Patient Data")
            summary_input = gr.Textbox(label="Clinical Summary", lines=5, value="A 55-year-old male with a history of headaches...")
            query_input = gr.Textbox(label="Your Query for the Agent", value="First, provide a qualitative overview of the changes between the two scans. Then, based on your initial findings, perform a precise quantitative analysis of the tumor volume progression.")
            
            gr.Markdown("**Initial/Follow-up Scans & Histology**")
            t1c1_input, t2f1_input = [gr.File(label="Initial T1c"), gr.File(label="Initial FLAIR")]
            t1c3_input, t2f3_input = [gr.File(label="Follow-up T1c"), gr.File(label="Follow-up FLAIR")]
            histo1_input = gr.File(label="Histology Slide (Optional)")
            
            btn = gr.Button("Run Unified Agent", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Unified Agent Report")
            output_report = gr.Textbox(label="Agent's Comprehensive Report", lines=25, interactive=False)

    def gradio_interface_handler(summary, query, t1c1, t2f1, t1c3, t2f3, histo1):
        files = {'t1c1': t1c1, 't2f1': t2f1, 't1c3': t1c3, 't2f3': t2f3, 'histo1': histo1}
        uploaded_files = {k: v for k, v in files.items() if v is not None}
        return run_unified_agent_flow(summary, query, uploaded_files)

    btn.click(gradio_interface_handler,
              inputs=[summary_input, query_input, t1c1_input, t2f1_input, t1c3_input, t2f3_input, histo1_input],
              outputs=[output_report])

# --- 7. Launch the App ---
if __name__ == "__main__":
    demo.launch(debug=True)