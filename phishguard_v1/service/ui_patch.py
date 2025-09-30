"""兼容旧版 gradio.DataFrame 构造参数"""
import gradio as gr

__all__ = ["DataFrame"]

def DataFrame(*args, height=None, **kwargs):
    """兼容不同版本 DataFrame：忽略 height 参数"""
    if height is not None and "height" not in gr.DataFrame.__init__.__code__.co_varnames:
        return gr.DataFrame(*args, **kwargs)
    if height is not None:
        kwargs.setdefault("elem_id", None)
        try:
            return gr.DataFrame(*args, height=height, **kwargs)
        except TypeError:
            pass
    return gr.DataFrame(*args, **kwargs)
