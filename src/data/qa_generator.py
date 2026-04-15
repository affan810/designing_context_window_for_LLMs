"""
QA Pair Generation.

Supports two modes:
  1. Rule-based: extracts simple factual QA pairs from sentences (no API calls).
  2. LLM-based (optional): uses OpenAI gpt-4o-mini for higher quality QA.

The rule-based mode is the default and requires no external API.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk

from src.utils.logging import get_logger

logger = get_logger(__name__)

import ssl as _ssl

def _nltk_download(resource: str) -> None:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        try:
            ctx = _ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            nltk.download(resource, quiet=True)
        except Exception:
            pass

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    _nltk_download("punkt_tab")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    _nltk_download("averaged_perceptron_tagger_eng")


# ---------------------------------------------------------------------------
# Rule-based QA generation
# ---------------------------------------------------------------------------

# Simple patterns: "X is/was/are/were Y" → Q: "What is/was X?" A: "Y"
_IS_PATTERN = re.compile(
    r"^([A-Z][^,\.]{2,60}?)\s+(is|was|are|were|became|remained)\s+(.{5,120})[\.!?]?$",
    re.IGNORECASE,
)

# "In YEAR, X did Y" → Q: "What did X do in YEAR?" A: "Y"
_IN_YEAR_PATTERN = re.compile(
    r"^In\s+(\d{4}),\s+(.{4,60}?)\s+(did|created|founded|built|wrote|discovered|invented|established)\s+(.{5,120})[\.!?]?$",
    re.IGNORECASE,
)


def _rule_based_qa_from_sentence(sentence: str) -> Optional[Tuple[str, str]]:
    """Try to extract a QA pair from a single sentence using simple patterns."""
    s = sentence.strip()

    m = _IS_PATTERN.match(s)
    if m:
        subject, verb, obj = m.group(1), m.group(2), m.group(3)
        question = f"What {verb} {subject.strip()}?"
        answer = obj.strip().rstrip(".,;")
        return question, answer

    m = _IN_YEAR_PATTERN.match(s)
    if m:
        year, subject, verb, obj = m.group(1), m.group(2), m.group(3), m.group(4)
        question = f"What did {subject.strip()} {verb} in {year}?"
        answer = obj.strip().rstrip(".,;")
        return question, answer

    return None


def _generate_who_what_questions(sentences: List[str]) -> List[Dict[str, str]]:
    """
    Broader rule-based extraction: create questions by swapping named entities/
    key noun phrases into question templates.
    """
    pairs: List[Dict[str, str]] = []

    for sentence in sentences:
        result = _rule_based_qa_from_sentence(sentence)
        if result:
            pairs.append({"question": result[0], "answer": result[1], "source_sentence": sentence})

    return pairs


def generate_factual_questions(story: str, min_pairs: int = 3) -> List[Dict[str, str]]:
    """
    Generate factual QA pairs from a story using rule-based extraction.

    Falls back to keyword-anchor questions if not enough patterns match.
    """
    sentences = nltk.sent_tokenize(story)
    pairs = _generate_who_what_questions(sentences)

    # Fallback: keyword-anchor questions for sentences with proper nouns
    if len(pairs) < min_pairs:
        for sent in sentences:
            if len(pairs) >= min_pairs * 2:
                break
            # Look for sentences with capitalized sequences (likely named entities)
            caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sent)
            if caps:
                entity = caps[0]
                # Simple "what" question anchored on the entity
                clean = sent.rstrip(".,;!?")
                q = f"What is mentioned about {entity} in the text?"
                if not any(p["answer"] == clean for p in pairs):
                    pairs.append({
                        "question": q,
                        "answer": clean,
                        "source_sentence": sent,
                    })

    return pairs


# ---------------------------------------------------------------------------
# Synthetic story + QA dataset builder
# ---------------------------------------------------------------------------

SYNTHETIC_STORIES = [
    {
        "story": (
            "The Amazon rainforest is the world's largest tropical rainforest, "
            "covering over 5.5 million square kilometres. It represents over half "
            "of the planet's remaining rainforests and comprises the largest and "
            "most biodiverse tract of tropical rainforest in the world. "
            "The Amazon River is the second longest river in the world by length "
            "but carries the greatest volume of water of any river system on Earth. "
            "In 2019, widespread fires burned large portions of the Amazon, drawing "
            "international attention to deforestation. Scientists estimate that the "
            "rainforest is home to 10% of all species on Earth."
        ),
        "qa_pairs": [
            {"question": "What is the Amazon rainforest?", "answer": "the world's largest tropical rainforest"},
            {"question": "How large is the Amazon rainforest?", "answer": "over 5.5 million square kilometres"},
            {"question": "What happened to the Amazon in 2019?", "answer": "widespread fires burned large portions of the Amazon"},
            {"question": "What percentage of Earth's species live in the rainforest?", "answer": "10%"},
        ],
    },
    {
        "story": (
            "Marie Curie was a Polish-born physicist and chemist who conducted "
            "pioneering research on radioactivity. She was the first woman to win a "
            "Nobel Prize and is the only person to have won Nobel Prizes in two "
            "different sciences: Physics in 1903 and Chemistry in 1911. "
            "Curie was born in Warsaw in 1867 and moved to Paris to pursue her studies. "
            "She discovered two elements: polonium, named after her homeland Poland, "
            "and radium. During World War I, she developed mobile X-ray units, "
            "nicknamed 'petites Curies', to help treat wounded soldiers. "
            "Marie Curie died in 1934 from aplastic anaemia, likely caused by her "
            "prolonged exposure to radiation."
        ),
        "qa_pairs": [
            {"question": "Who was Marie Curie?", "answer": "a Polish-born physicist and chemist who conducted pioneering research on radioactivity"},
            {"question": "How many Nobel Prizes did Marie Curie win?", "answer": "two Nobel Prizes"},
            {"question": "What elements did Marie Curie discover?", "answer": "polonium and radium"},
            {"question": "When was Marie Curie born?", "answer": "1867"},
            {"question": "What was polonium named after?", "answer": "her homeland Poland"},
            {"question": "When did Marie Curie die?", "answer": "1934"},
        ],
    },
    {
        "story": (
            "The Internet is a global network of interconnected computers that "
            "communicate using the Internet Protocol Suite (TCP/IP). It was developed "
            "in the late 1960s as ARPANET, a project funded by the United States "
            "Department of Defense. Tim Berners-Lee invented the World Wide Web in 1989 "
            "while working at CERN in Switzerland. The Web is a service built on top of "
            "the Internet, providing a system of interlinked hypertext documents. "
            "Today, the Internet connects over 5 billion users worldwide. "
            "The first email was sent by Ray Tomlinson in 1971. "
            "Google was founded in 1998 by Larry Page and Sergey Brin as a search engine "
            "to help users navigate the rapidly growing web."
        ),
        "qa_pairs": [
            {"question": "What is the Internet?", "answer": "a global network of interconnected computers"},
            {"question": "When was the World Wide Web invented?", "answer": "1989"},
            {"question": "Who invented the World Wide Web?", "answer": "Tim Berners-Lee"},
            {"question": "When was the first email sent?", "answer": "1971"},
            {"question": "Who sent the first email?", "answer": "Ray Tomlinson"},
            {"question": "When was Google founded?", "answer": "1998"},
        ],
    },
    {
        "story": (
            "The Great Wall of China is a series of fortifications built across the "
            "historical northern borders of ancient Chinese states and Imperial China. "
            "Construction began as early as the 7th century BC and continued through "
            "the Ming dynasty (1368–1644 AD). The wall stretches over 21,196 kilometres "
            "according to a 2012 archaeological survey. It was built to protect Chinese "
            "states from the raids and invasions of various nomadic groups. "
            "The Ming dynasty built the most well-preserved section of the wall "
            "near Beijing. The Great Wall was designated a UNESCO World Heritage Site "
            "in 1987. Despite popular belief, the Great Wall is not visible from space "
            "with the naked eye."
        ),
        "qa_pairs": [
            {"question": "What is the Great Wall of China?", "answer": "a series of fortifications built across the historical northern borders of ancient Chinese states"},
            {"question": "How long is the Great Wall of China?", "answer": "over 21,196 kilometres"},
            {"question": "When was the Great Wall designated a UNESCO World Heritage Site?", "answer": "1987"},
            {"question": "Why was the Great Wall built?", "answer": "to protect Chinese states from the raids and invasions of various nomadic groups"},
            {"question": "Which dynasty built the most well-preserved section?", "answer": "The Ming dynasty"},
        ],
    },
    {
        "story": (
            "William Shakespeare was an English playwright, poet, and actor, widely "
            "regarded as the greatest writer in the English language. He was born in "
            "Stratford-upon-Avon in 1564 and died in 1616. Shakespeare wrote 37 plays "
            "and 154 sonnets during his lifetime. His works have been translated into "
            "every major language and are performed more often than those of any other "
            "playwright. Shakespeare is often called England's national poet and the "
            "'Bard of Avon'. His plays include Hamlet, Othello, King Lear, Macbeth, "
            "and Romeo and Juliet. The Globe Theatre was built in 1599 by Shakespeare's "
            "playing company, the Lord Chamberlain's Men. Many of Shakespeare's works "
            "explore themes of love, jealousy, power, and fate."
        ),
        "qa_pairs": [
            {"question": "Who was William Shakespeare?", "answer": "an English playwright, poet, and actor"},
            {"question": "Where was Shakespeare born?", "answer": "Stratford-upon-Avon"},
            {"question": "When was Shakespeare born?", "answer": "1564"},
            {"question": "How many plays did Shakespeare write?", "answer": "37 plays"},
            {"question": "When was the Globe Theatre built?", "answer": "1599"},
            {"question": "What are some of Shakespeare's famous plays?", "answer": "Hamlet, Othello, King Lear, Macbeth, and Romeo and Juliet"},
        ],
    },
    {
        "story": (
            "The human brain is the command center of the human nervous system. "
            "It weighs about 1.4 kilograms and contains approximately 86 billion neurons. "
            "The brain is divided into three main parts: the cerebrum, cerebellum, and "
            "brainstem. The cerebrum is the largest part and is responsible for higher "
            "brain functions such as thought, memory, and speech. The cerebellum controls "
            "balance, coordination, and fine motor movements. The brainstem connects the "
            "brain to the spinal cord and controls basic life functions. "
            "The brain uses about 20% of the body's total energy despite making up only "
            "2% of body weight. Neurons communicate through synapses using chemical "
            "messengers called neurotransmitters."
        ),
        "qa_pairs": [
            {"question": "What is the human brain?", "answer": "the command center of the human nervous system"},
            {"question": "How much does the human brain weigh?", "answer": "about 1.4 kilograms"},
            {"question": "How many neurons does the brain contain?", "answer": "approximately 86 billion neurons"},
            {"question": "What are the three main parts of the brain?", "answer": "the cerebrum, cerebellum, and brainstem"},
            {"question": "What percentage of body energy does the brain use?", "answer": "about 20%"},
        ],
    },
    {
        "story": (
            "The Python programming language was created by Guido van Rossum and first "
            "released in 1991. Python emphasizes code readability and simplicity, using "
            "significant indentation as a core syntactic feature. It supports multiple "
            "programming paradigms including procedural, object-oriented, and functional "
            "programming. Python is widely used in web development, data science, "
            "artificial intelligence, and scientific computing. The Python Software "
            "Foundation manages the language and its development. Python 3.0 was "
            "released in 2008 and introduced major changes that were not backward "
            "compatible with Python 2. As of 2024, Python consistently ranks as one "
            "of the most popular programming languages worldwide."
        ),
        "qa_pairs": [
            {"question": "Who created Python?", "answer": "Guido van Rossum"},
            {"question": "When was Python first released?", "answer": "1991"},
            {"question": "When was Python 3.0 released?", "answer": "2008"},
            {"question": "What does Python emphasize?", "answer": "code readability and simplicity"},
            {"question": "What manages Python's development?", "answer": "The Python Software Foundation"},
        ],
    },
    {
        "story": (
            "Climate change refers to long-term shifts in global temperatures and "
            "weather patterns. While some climate change is natural, since the mid-20th "
            "century, human activities have been the main driver. Burning fossil fuels "
            "releases greenhouse gases like carbon dioxide and methane into the atmosphere. "
            "The Intergovernmental Panel on Climate Change (IPCC) was established in 1988 "
            "to assess the scientific basis of climate change. Global average temperatures "
            "have risen by approximately 1.1°C since the pre-industrial period. "
            "The Paris Agreement, adopted in 2015, aims to limit global warming to "
            "1.5°C above pre-industrial levels. Rising sea levels, more frequent extreme "
            "weather events, and biodiversity loss are among the projected impacts."
        ),
        "qa_pairs": [
            {"question": "What is climate change?", "answer": "long-term shifts in global temperatures and weather patterns"},
            {"question": "When was the IPCC established?", "answer": "1988"},
            {"question": "What is the Paris Agreement?", "answer": "an agreement adopted in 2015 that aims to limit global warming to 1.5°C above pre-industrial levels"},
            {"question": "By how much have global temperatures risen?", "answer": "approximately 1.1°C since the pre-industrial period"},
        ],
    },
]


def build_synthetic_dataset(output_path: str) -> List[Dict]:
    """Save the synthetic dataset to disk and return it."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(SYNTHETIC_STORIES, f, indent=2)
    logger.info(f"Saved {len(SYNTHETIC_STORIES)} synthetic stories → {output_path}")
    return SYNTHETIC_STORIES


# ---------------------------------------------------------------------------
# Optional: LLM-based QA generation (gpt-4o-mini)
# ---------------------------------------------------------------------------

def generate_qa_with_llm(story: str, api_key: str, num_pairs: int = 5) -> List[Dict[str, str]]:
    """
    Use OpenAI gpt-4o-mini to generate QA pairs from a story.
    Only call this when rule-based generation is insufficient.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    client = OpenAI(api_key=api_key)
    prompt = (
        f"Read the following story and generate exactly {num_pairs} factual "
        "question-answer pairs in JSON format.\n\n"
        f"Story:\n{story}\n\n"
        "Return a JSON array of objects with keys 'question' and 'answer'. "
        "Answers should be short (1-2 sentences) and directly from the text."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\n?", "", raw).rstrip("```").strip()
    pairs = json.loads(raw)
    return pairs


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/processed/dataset.json"
    build_synthetic_dataset(out)
    print(f"Dataset ready at {out}")
