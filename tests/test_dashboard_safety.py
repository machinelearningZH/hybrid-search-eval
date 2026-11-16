import json
from html.parser import HTMLParser
from pathlib import Path

from _core.utils import create_html_dashboard


class DashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.scripts: list[str] = []
        self.in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self.in_script = True
            self.scripts.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_script = False

    def handle_data(self, data: str) -> None:
        if self.in_script:
            self.scripts[-1] += data


def test_dashboard_treats_names_and_metric_labels_as_text(tmp_path: Path) -> None:
    project = '<img id="project-injected" src=x onerror="alert(1)">'
    model = '</script><img id="model-injected" src=x onerror="alert(1)">'
    short_name = '<svg onload="alert(1)">& model'
    metric = 'mrr@10" onclick="alert(1)'
    result = {
        "model": model,
        "model_short": short_name,
        "alpha": 1.0,
        metric: 0.8,
        "avg_embed_time_ms": 2.0,
        "total_embed_time_ms": 20.0,
        "num_queries": 2,
        "num_documents": 10,
    }

    create_html_dashboard(
        [result],
        {},
        tmp_path,
        "20260101_120000",
        {"project_id": project, "visualization": {"pareto_quality_metric": metric}},
    )

    dashboard = (tmp_path / "dashboard_20260101_120000.html").read_text()
    parsed = DashboardParser()
    parsed.feed(dashboard)
    assert all(tag not in {"img", "svg"} for tag, _ in parsed.tags)
    assert all("onclick" not in attrs for _, attrs in parsed.tags)
    assert len(parsed.scripts) == 2
    script = parsed.scripts[-1]
    rows = json.loads(script.split("const resultsData = ", 1)[1].split(";\n", 1)[0])
    assert rows[0]["model"] == model
    assert rows[0]["model_short"] == short_name
    assert rows[0][metric] == 0.8
    assert any(attrs.get("data-sort") == metric for _, attrs in parsed.tags)
    assert "${row.model}" not in script
    assert "${row.model_short}" not in script
