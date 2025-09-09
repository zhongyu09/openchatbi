# Custom CSS for styling the chat interface
custom_css = """
#chatbot {
    height: 600px !important;
    font-family: "Inter", "Helvetica Neue", sans-serif;
}
#chatbot .wrap.svelte-1cl84sx {
    background: #f5f7fa;
    border-radius: 12px;
    padding: 8px;
}
.message.user {
    background-color: #d1e9ff !important;
    border-radius: 12px 12px 0 12px;
    margin: 4px 0;
    padding: 10px 14px;
    font-size: 15px;
}
.message.bot {
    background-color: #ffffff !important;
    border-radius: 12px 12px 12px 0;
    margin: 4px 0;
    padding: 10px 14px;
    font-size: 15px;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.08);
}
#msg {
    font-family: "Inter", "Helvetica Neue", sans-serif;
    font-size: 15px;
}
#description {
    font-family: "Inter", "Helvetica Neue", sans-serif;
    font-size: 14px;
    color: #374151;
    line-height: 1.6;
}
"""
