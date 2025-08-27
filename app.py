import validators
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv
import os
load_dotenv()

# Get your NVIDIA API key from environment variable (or hardcode for testing)
api_key = os.getenv("NVIDIA_KEY")
if not api_key:
    raise ValueError("Please set NVIDIA_KEY in your environment variables")

# Initialize the LLM
llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=api_key,
    temperature=0.2,
    top_p=0.7,
    max_completion_tokens=1024,
)

# Prompt template
prmpt_template = """
Summarize the following content in **Markdown format** with clear structure:
- Top-level heading: # Content Summary
- Bullet points for key points
- Subheadings where necessary
- Keep it concise and easy to read
Content: {text}
"""
prompt = PromptTemplate(template=prmpt_template, input_variables=["text"])

# Function to summarize URL content
def summarize_content(generic_url):
    if not generic_url.strip():
        return "Please provide a URL"

    if not validators.url(generic_url):
        return "Please enter a valid URL (YouTube or Website)"

    try:
        # Load data
        if "youtube.com" in generic_url:
            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
        else:
            loader = UnstructuredURLLoader(
                urls=[generic_url],
                ssl_verify=False,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/116.0.0.0 Safari/537.36"
                    )
                },
            )
        docs = loader.load()

        # Summarization chain
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        output_summary = chain.invoke(docs)

        # Return Markdown-formatted summary
        if isinstance(output_summary, dict) and "output_text" in output_summary:
            return output_summary["output_text"]
        else:
            return str(output_summary)

    except Exception as e:
        return f"Exception: {e}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## NVIDIA LLM: Summarize Text from YouTube or Websites")

    with gr.Row():
        url_input = gr.Textbox(
            label="URL", placeholder="Enter YouTube or Website URL here..."
        )

    summarize_btn = gr.Button("Summarize")
    output_md = gr.Markdown()

    summarize_btn.click(
        summarize_content,
        inputs=[url_input],
        outputs=[output_md]
    )

if __name__ == "__main__":
    demo.launch()
