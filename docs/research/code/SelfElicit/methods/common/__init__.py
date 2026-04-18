from .classify import classify_sentences
from .context import update_context
from utils.utils import inverse_softmax


def preprocess(sentences, model, sample_id, cls_cache, context_cache, language):
    # knowledgable selection
    is_core_sentences = classify_sentences(sentences,
                                           model=model,
                                           sample_id=sample_id,
                                           cls_cache=cls_cache,
                                           language=language)

    # decontextualisation
    sentences = update_context(sentences,
                               is_core_sentences,
                               model=model,
                               sample_id=sample_id,
                               context_cache=context_cache,
                               language=language)

    return sentences, is_core_sentences


def get_default_result():
    prob = inverse_softmax(0.)
    response = f"No knowledge"

    return prob, response
