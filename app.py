import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import re
import traceback
from urllib.parse import quote_plus

import requests
import streamlit as st
import wikipediaapi


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

APP_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = APP_DIR / "Autonomous-AI-Research-Assistant.ipynb"

wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="AIResearchAssistant/1.0 (your.email@example.com)",
)

BACKEND_CELL_MARKERS = (
    "wiki_wiki = wikipediaapi.Wikipedia",
    "def search_semantic_scholar",
    "def retrieve_and_summarize",
    "def answer_question",
)

REQUIRED_BACKEND_NAMES = (
    "search_semantic_scholar",
    "retrieve_and_summarize",
    "answer_question",
    "combined_answer",
    "get_wikipedia_summary_and_url",
    "summarizer",
)


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 8px;
        }

        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 8px;
            font-weight: 650;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def _source_from_cell(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)

    cleaned_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def search_semantic_scholar(query: str, limit: int = 10) -> list[dict]:
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={quote_plus(query)}&limit={limit}&fields=title,abstract,url,authors,year"
    )
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return response.json().get("data", [])
    return []


def retrieve_and_summarize(query: str, limit: int = 10) -> list[dict]:
    papers = search_semantic_scholar(query, limit)
    results = []
    for paper in papers:
        abstract = paper.get("abstract") or ""
        summary = abstract[:250] + "..." if len(abstract) > 250 else abstract
        if not summary:
            summary = "Abstract not available"
        results.append(
            {
                "title": paper.get("title", "No Title"),
                "summary": summary,
                "url": paper.get("url", "URL not available"),
                "abstract": abstract,
            }
        )
    return results


def answer_question(question: str, context: str) -> str:
    if not context:
        return "No context available for this paper."

    question_terms = [term for term in re.findall(r"[A-Za-z0-9-]+", question.lower()) if len(term) > 2]
    context_lower = context.lower()
    if any(term in context_lower for term in question_terms):
        return context[:400] + ("..." if len(context) > 400 else "")

    return context[:200] + ("..." if len(context) > 200 else "")


def get_wikipedia_summary_and_url(query: str) -> tuple[str | None, str | None]:
    title = query.strip()
    page = wiki_wiki.page(title)
    if not page.exists():
        search_title = title.split()[-1].capitalize()
        page = wiki_wiki.page(search_title)
    if not page.exists():
        return None, None
    return page.summary[:1000], page.fullurl


def combined_answer(question: str, paper_abstract: str) -> str:
    answer = answer_question(question, paper_abstract)
    if (
        answer.lower() in ["no answer found.", "no context available for this paper.", "", "n/a"]
        or len(answer) < 15
        or "www." in answer
        or ".org" in answer
        or ".com" in answer
        or ".net" in answer
    ):
        wiki_summary, wiki_url = get_wikipedia_summary_and_url(question)
        if wiki_summary:
            return f"Wikipedia Summary:\n{wiki_summary}\n\nRead more at: {wiki_url}"
        return "Sorry, no answer found in paper or Wikipedia."
    return f"Paper-based Answer:\n{answer}"


@st.cache_resource(show_spinner=False)
def load_backend() -> dict:
    return {
        "search_semantic_scholar": search_semantic_scholar,
        "retrieve_and_summarize": retrieve_and_summarize,
        "answer_question": answer_question,
        "combined_answer": combined_answer,
        "get_wikipedia_summary_and_url": get_wikipedia_summary_and_url,
        "summarizer": None,
    }


@contextlib.contextmanager
def capture_backend_output():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


def clean_text(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    text = " ".join(str(value).split())
    return text or fallback


def interpret_user_input(user_input: str) -> str:
    text = clean_text(user_input, "")
    if not text:
        return ""

    interpreted = re.sub(r"[?!.]+$", "", text).strip()
    patterns = [
        r"^(please\s+)?(can|could|would)\s+you\s+(please\s+)?",
        r"^(please\s+)?(tell me|explain|describe)\s+(about|on)?\s*",
        r"^(please\s+)?(research|find|search)\s+(about|on|for)?\s*",
        r"^(please\s+)?(give|provide|show)\s+(me\s+)?(a\s+|an\s+)?(research\s+)?(report|summary|overview)\s+(about|on|for)\s+",
        r"^i\s+(want|need|would like)\s+to\s+(know|learn|understand|research|find out)\s+(about|on)?\s*",
        r"^(what|which)\s+(is|are|was|were)\s+(the\s+)?",
        r"^how\s+(does|do|can|could|is|are)\s+(the\s+)?",
        r"^why\s+(is|are|does|do)\s+(the\s+)?",
    ]

    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            updated = re.sub(pattern, "", interpreted, flags=re.IGNORECASE).strip()
            if updated != interpreted:
                interpreted = updated
                changed = True

    interpreted = re.sub(r"\bhelp(s|ed|ing)?\s+(with|in|for)\b", "for", interpreted, flags=re.IGNORECASE)
    interpreted = re.sub(r"\b(information|info|details)\s+(about|on)\s+", "", interpreted, flags=re.IGNORECASE)
    interpreted = re.sub(r"\s+", " ", interpreted).strip(" ,:;-")

    if len(interpreted) < 3:
        return text
    return interpreted


def expand_common_terms(text: str) -> str:
    replacements = (
        (r"\bai\b", "artificial intelligence"),
        (r"\bllms\b", "large language models"),
        (r"\bllm\b", "large language model"),
        (r"\bnlp\b", "natural language processing"),
        (r"\bml\b", "machine learning"),
    )

    expanded = text
    for pattern, replacement in replacements:
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def build_query_variants(topic: str) -> list[str]:
    normalized = " ".join(topic.split())
    variants = [normalized]

    expanded_terms = expand_common_terms(normalized)
    if expanded_terms != normalized:
        variants.append(expanded_terms)

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "with",
    }
    keyword_query = " ".join(
        word for word in re.findall(r"[A-Za-z0-9-]+", expanded_terms) if word.lower() not in stop_words
    )
    if keyword_query:
        variants.append(keyword_query)

    lower_topic = normalized.lower()
    if "agent" in lower_topic and "literature review" in lower_topic:
        variants.extend(
            [
                "autonomous agents literature review",
                "large language models literature review",
                "artificial intelligence scientific literature review",
            ]
        )

    variants.append(quote_plus(normalized))

    unique_variants = []
    for variant in variants:
        if variant and variant not in unique_variants:
            unique_variants.append(variant)
    return unique_variants


def build_wikipedia_queries(topic: str) -> list[str]:
    normalized = " ".join(topic.split())
    expanded_terms = expand_common_terms(normalized)
    lower_topic = expanded_terms.lower()
    candidates = []

    if "systematic" in lower_topic and ("review" in lower_topic or "literature" in lower_topic):
        candidates.extend(["Systematic review", "Literature review"])

    if "literature review" in lower_topic or "scientific literature" in lower_topic:
        candidates.extend(["Literature review", "Scientific literature"])

    if "artificial intelligence" in lower_topic:
        candidates.extend(["Artificial intelligence", "Applications of artificial intelligence"])

    if "large language model" in lower_topic:
        candidates.append("Large language model")

    if "agent" in lower_topic:
        candidates.extend(["Intelligent agent", "Software agent"])

    candidates.extend([normalized, expanded_terms])

    unique_candidates = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def retrieve_wikipedia_fallback(topic: str, backend: dict) -> tuple[list[dict], list[str], str]:
    attempted_queries = build_wikipedia_queries(topic)

    for query in attempted_queries:
        summary, url = backend["get_wikipedia_summary_and_url"](query)
        if summary and url:
            return (
                [
                    {
                        "title": f"Wikipedia: {query}",
                        "summary": clean_text(summary, "No summary available."),
                        "url": url,
                        "abstract": summary,
                        "source_type": "Wikipedia",
                    }
                ],
                attempted_queries,
                query,
            )

    return [], attempted_queries, ""


def generate_overview(topic: str, results: list[dict], backend: dict) -> str:
    summaries = [
        clean_text(result.get("summary"), "")
        for result in results
        if clean_text(result.get("summary"), "").lower() != "abstract not available"
    ]

    if not summaries:
        return (
            f"The search returned sources for {topic}, but their abstracts were not available."
        )

    # Just combine the available summaries
    return " ".join(summaries[:3])


def build_report(
    topic: str,
    results: list[dict],
    backend: dict,
    search_query: str | None = None,
    original_input: str | None = None,
) -> str:
    generated_on = dt.date.today().isoformat()

    if not results:
        lines = [
            f"# Research Report: {topic}",
            "",
            f"Generated on {generated_on}.",
        ]
        if original_input and original_input != topic:
            lines.extend(["", f"Original input: {original_input}"])
        lines.extend(["", "No Semantic Scholar or Wikipedia fallback sources were found for this topic."])
        return "\n".join(lines)

    overview = generate_overview(topic, results, backend)
    source_types = sorted({clean_text(result.get("source_type"), "Semantic Scholar") for result in results})
    source_type_text = ", ".join(source_types)
    lines = [
        f"# Research Report: {topic}",
        "",
        f"Generated on {generated_on} from {len(results)} source(s): {source_type_text}.",
    ]
    if original_input and original_input != topic:
        lines.extend(["", f"Original input: {original_input}"])
    if search_query and search_query != topic:
        lines.extend(["", f"Search query used: {search_query}"])

    lines.extend(
        [
            "",
            "## Executive Summary",
            overview,
            "",
            "## Source Summaries",
        ]
    )

    for index, result in enumerate(results, start=1):
        title = clean_text(result.get("title"), "Untitled source")
        summary = clean_text(result.get("summary"), "No summary available.")
        url = clean_text(result.get("url"), "")
        source_type = clean_text(result.get("source_type"), "Semantic Scholar")

        lines.extend(["", f"### {index}. {title}", f"Source type: {source_type}", summary])
        if url and url.lower() != "url not available":
            lines.append(f"Source: {url}")

    lines.extend(["", "## Sources"])
    for index, result in enumerate(results, start=1):
        title = clean_text(result.get("title"), "Untitled source")
        url = clean_text(result.get("url"), "URL not available")
        source_type = clean_text(result.get("source_type"), "Semantic Scholar")
        lines.append(f"{index}. [{source_type}] {title} - {url}")

    return "\n".join(lines)


def run_research(original_input: str, topic: str, limit: int) -> tuple[list[dict], str, str, list[str], str]:
    progress = st.progress(0, text="Preparing research workflow")
    query_variants = build_query_variants(topic)
    tried_queries = list(query_variants)
    all_logs = []
    results = []
    used_query = topic

    with st.status("Research in progress", expanded=True) as status:
        st.write("Loading backend functions")
        backend = load_backend()
        progress.progress(20, text="Backend loaded")

        for index, query in enumerate(query_variants, start=1):
            st.write(f"Searching Semantic Scholar: {query}")
            progress_value = min(75, 25 + int(45 * index / len(query_variants)))
            progress.progress(progress_value, text=f"Searching source query {index} of {len(query_variants)}")

            with capture_backend_output() as backend_logs:
                results = backend["retrieve_and_summarize"](query, limit=limit)

            logs = backend_logs.getvalue().strip()
            if logs:
                all_logs.append(f"Query: {query}\n{logs}")

            if results:
                used_query = query
                st.write(f"Found {len(results)} source(s)")
                break

        if not results:
            st.write("No Semantic Scholar papers found. Checking Wikipedia fallback")
            progress.progress(75, text="Checking Wikipedia fallback")
            try:
                wiki_results, wiki_queries, wiki_query = retrieve_wikipedia_fallback(topic, backend)
            except Exception as exc:
                wiki_results = []
                wiki_queries = build_wikipedia_queries(topic)
                wiki_query = ""
                all_logs.append(f"Wikipedia fallback failed: {exc}")
            tried_queries.extend([f"Wikipedia: {query}" for query in wiki_queries])
            if wiki_queries:
                all_logs.append("Wikipedia fallback queries tried:\n" + "\n".join(wiki_queries))
            if wiki_results:
                results = wiki_results
                used_query = wiki_query
                st.write(f"Using Wikipedia fallback: {wiki_query}")

        progress.progress(80, text="Building final report")
        st.write("Generating final report")
        report = build_report(topic, results, backend, search_query=used_query, original_input=original_input)

        progress.progress(100, text="Research complete")
        status.update(label="Research complete", state="complete", expanded=False)

    return results, report, "\n\n".join(all_logs), tried_queries, used_query


def render_sources(results: list[dict]) -> None:
    if not results:
        st.warning("No sources were found. Try a shorter keyword-style topic.")
        return

    st.subheader("Sources Found")
    st.caption(f"{len(results)} source(s) returned by the backend workflow.")

    for index, result in enumerate(results, start=1):
        title = clean_text(result.get("title"), "Untitled source")
        summary = clean_text(result.get("summary"), "No summary available.")
        url = clean_text(result.get("url"), "")
        source_type = clean_text(result.get("source_type"), "Semantic Scholar")

        with st.expander(f"{index}. {title}", expanded=index == 1):
            st.caption(source_type)
            st.write(summary)
            if url and url.lower() != "url not available":
                st.markdown(f"[Open source]({url})")


def render_report(topic: str, report: str) -> None:
    st.subheader("Final Generated Report")
    st.markdown(report)
    st.download_button(
        "Download Report",
        data=report,
        file_name=f"{topic.strip().lower().replace(' ', '-')[:60] or 'research'}-report.md",
        mime="text/markdown",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Autonomous AI Research Assistant",
        layout="wide",
    )
    apply_page_style()

    st.title("Autonomous AI Research Assistant")

    with st.sidebar:
        st.header("Research Settings")
        limit = st.slider("Sources", min_value=1, max_value=10, value=5)

    with st.form("research_form", clear_on_submit=False):
        user_input = st.text_input(
            "Research topic or question",
            placeholder="Example: how can AI tools help with systematic literature reviews?",
        )
        submitted = st.form_submit_button("Start Research", type="primary", use_container_width=True)

    if submitted:
        normalized_input = user_input.strip()
        interpreted_topic = interpret_user_input(normalized_input)
        if not interpreted_topic:
            st.error("Enter a research topic or question before starting.")
            return

        try:
            results, report, logs, tried_queries, used_query = run_research(
                normalized_input,
                interpreted_topic,
                limit,
            )
            st.session_state["last_research"] = {
                "original_input": normalized_input,
                "topic": interpreted_topic,
                "results": results,
                "report": report,
                "logs": logs,
                "tried_queries": tried_queries,
                "used_query": used_query,
            }
        except Exception as exc:
            st.error("Research could not be completed.")
            st.info(str(exc))
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            return

    last_research = st.session_state.get("last_research")
    if last_research:
        source_count = len(last_research["results"])
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.caption("Interpreted Research Topic")
            st.markdown(f"**{last_research['topic']}**")
            if last_research.get("original_input") and last_research["original_input"] != last_research["topic"]:
                st.caption(f"Original input: {last_research['original_input']}")
        col_b.metric("Sources Found", source_count)

        render_sources(last_research["results"])
        if not last_research["results"]:
            with st.expander("Queries tried", expanded=True):
                for query in last_research.get("tried_queries", []):
                    st.write(f"- {query}")
        render_report(last_research["topic"], last_research["report"])

        if last_research.get("logs"):
            with st.expander("Backend Logs", expanded=not last_research["results"]):
                st.code(last_research["logs"])


if __name__ == "__main__":
    main()
