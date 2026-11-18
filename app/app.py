import gradio as gr
from maliki_rag.pipeline import Pipeline

pl = Pipeline()
CSS = ".prose, .md {direction: rtl; text-align: right;} textarea {direction: rtl;}"


def respond(q: str) -> str:
    if not q.strip():
        return "اكتب سؤالك."
    return pl.ask(q)["markdown"]


with gr.Blocks(css=CSS, title="البحث في كتب المذهب المالكي") as demo:
    gr.Markdown("# البحث في كتب المذهب المالكي\nيجيب من النصوص المدخلة فقط مع الإحالة. ليس فتوى.")
    q = gr.Textbox(label="السؤال", lines=2, rtl=True)
    btn = gr.Button("ابحث")
    out = gr.Markdown(rtl=True)
    btn.click(respond, q, out)
    q.submit(respond, q, out)

if __name__ == "__main__":
    demo.launch()
