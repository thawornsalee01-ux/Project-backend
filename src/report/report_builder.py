# src/report/report_builder.py

import json
from pathlib import Path
from typing import List, Iterable, Optional
from datetime import datetime
from html import escape
from difflib import SequenceMatcher

from src.diff.diff import Change


class ReportBuilder:
    def __init__(self, output_dir: str = "data/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- JSON ---------------- #
    def save_json(
        self,
        doc_name: str,
        v1_label: str,
        v2_label: str,
        changes: List[Change],
    ) -> Path:
        data = {
            "document_name": doc_name,
            "version_old": v1_label,
            "version_new": v2_label,
            "changes": [
                {
                    "change_type": c.change_type,
                    "section_label": c.section_label,
                    "old_text": c.old_text,
                    "new_text": c.new_text,
                    "ai_comment": getattr(c, "ai_comment", None),
                    "ai_suggestion": getattr(c, "ai_suggestion", None),
                    "coverage": getattr(c, "coverage", None),
                    "mean_similarity": getattr(c, "mean_similarity", None),
                }
                for c in changes
            ],
            "generated_at": datetime.utcnow().isoformat(),
        }

        filename = f"{self._safe_name(doc_name)}_{v1_label}_vs_{v2_label}.json"
        out_path = self.output_dir / filename
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path

    # ---------------- helper สำหรับ diff ---------------- #
    def _tokens(self, text: str) -> List[str]:
        parts: List[str] = []
        for line in text.split("\n"):
            if line:
                parts.extend(line.split(" "))
            parts.append("\n")
        if parts and parts[-1] == "\n":
            parts.pop()
        return parts

    def _html_from_tokens(self, tokens: Iterable[str]) -> str:
        out_parts: List[str] = []
        for tok in tokens:
            if tok == "\n":
                out_parts.append("<br>")
            else:
                out_parts.append(escape(tok))

        html = ""
        for i, part in enumerate(out_parts):
            if i > 0 and part != "<br>" and out_parts[i - 1] != "<br>":
                html += " "
            html += part
        return html

    def _highlight_old(self, old_text: Optional[str], new_text: Optional[str]) -> str:
        if not old_text:
            return ""

        if not new_text:
            return f'<span class="chg-del">{self._html_from_tokens(self._tokens(old_text))}</span>'

        old_tokens = self._tokens(old_text)
        new_tokens = self._tokens(new_text)

        sm = SequenceMatcher(None, old_tokens, new_tokens)
        pieces: List[str] = []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            chunk = old_tokens[i1:i2]
            if not chunk:
                continue

            if tag in ("delete", "replace"):
                pieces.append(
                    f'<span class="chg-del">{self._html_from_tokens(chunk)}</span>'
                )
            else:
                pieces.append(self._html_from_tokens(chunk))

        return "".join(pieces)

    def _highlight_new(self, old_text: Optional[str], new_text: Optional[str]) -> str:
        if not new_text:
            return ""

        if not old_text:
            return f'<span class="chg-add">{self._html_from_tokens(self._tokens(new_text))}</span>'

        old_tokens = self._tokens(old_text)
        new_tokens = self._tokens(new_text)

        sm = SequenceMatcher(None, old_tokens, new_tokens)
        pieces: List[str] = []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            chunk = new_tokens[j1:j2]
            if not chunk:
                continue

            if tag in ("insert", "replace"):
                pieces.append(
                    f'<span class="chg-add">{self._html_from_tokens(chunk)}</span>'
                )
            else:
                pieces.append(self._html_from_tokens(chunk))

        return "".join(pieces)

    # ---------------- HTML ---------------- #
    def save_html(
        self,
        doc_name: str,
        v1_label: str,
        v2_label: str,
        changes: List[Change],
        summary_text: str | None = None,
        overall_risk_level: str | None = None,
        edit_intensity: str | None = None,
    ) -> Path:

        rows_parts: list[str] = []

        for c in changes:
            old_html = self._highlight_old(c.old_text, c.new_text)
            new_html = self._highlight_new(c.old_text, c.new_text)

            ai_comment = escape(getattr(c, "ai_comment", "") or "-")
            ai_suggestion = escape(getattr(c, "ai_suggestion", "") or "-")

            row_class = f"type-{escape(c.change_type)}"

            # ❌ ตัด Risk ออก → ไม่มี risk_level / risk_comment แล้ว

            row = (
                f'<tr class="{row_class}">'
                f"<td>{old_html}</td>"
                f"<td>{new_html}</td>"
                f"<td>{ai_comment}</td>"
                f"<td>{ai_suggestion}</td>"
                f"<td>{escape(c.section_label or '')}</td>"
                f"<td>{escape(c.change_type)}</td>"
                "</tr>"
            )
            rows_parts.append(row)

        rows_html = "\n".join(rows_parts)

        safe_summary = escape(summary_text or "").replace("\n", "<br>")

        summary_block = (
            "<section>"
            "<h2>สรุปภาพรวม</h2>"
            f"<p>{safe_summary}</p>"
            "</section>"
        ) if summary_text else ""

        html = (
            "<!DOCTYPE html>"
            '<html lang="th">'
            "<head>"
            '<meta charset="utf-8" />'
            f"<title>Diff Report - {escape(doc_name)}</title>"
            "<style>"
            "body { font-family: sans-serif; margin: 20px; }"

            ".table-scroll { overflow-x: auto; max-width: 100%; }"

            "table { border-collapse: collapse; width: 100%; min-width: 1800px; table-layout: fixed; }"
            "th, td { border: 1px solid #2E2E2E; padding: 8px; vertical-align: top; }"
            "th { background-color: #C7C7C7; }"

            "th:nth-child(1), td:nth-child(1),"
            "th:nth-child(2), td:nth-child(2) {"
            "  width: 30%;"
            "  word-wrap: break-word;"
            "  white-space: normal;"
            "}"

            "th:nth-child(3), td:nth-child(3),"
            "th:nth-child(4), td:nth-child(4) {"
            "  width: 18%;"
            "  word-wrap: break-word;"
            "  white-space: normal;"
            "}"

            "th:nth-child(5), td:nth-child(5),"
            "th:nth-child(6), td:nth-child(6) {"
            "  width: 5%;"
            "  text-align: center;"
            "}"

            ".type-ADDED, .type-REMOVED, .type-MODIFIED { background-color: #FFFBED; }"
            ".chg-del { background-color: #FFA6A6; text-decoration: line-through; }"
            ".chg-add { background-color: #C7FFBF; }"

            "</style>"
            "</head>"
            "<body>"
            "<h1>Document Versioning Compare</h1>"
            f"<p><strong>Document:</strong> {escape(doc_name)}</p>"
            f"<p><strong>Compare:</strong> {escape(v1_label)} → {escape(v2_label)}</p>"
            f"{summary_block}"
            "<h2>รายละเอียดการเปลี่ยนแปลง</h2>"
            '<div class="table-scroll">'
            "<table>"
            "<thead><tr>"
            "<th>Old Text</th>"
            "<th>New Text</th>"
            "<th>AI Comment</th>"
            "<th>AI Suggestion</th>"
            "<th>Section</th>"
            "<th>Type</th>"
            "</tr></thead>"
            "<tbody>"
            f"{rows_html}"
            "</tbody>"
            "</table>"
            "</div>"
            "</body></html>"
        )

        filename = f"{self._safe_name(doc_name)}_{v1_label}_vs_{v2_label}.html"
        out_path = self.output_dir / filename
        out_path.write_text(html, encoding="utf-8")
        return out_path

    def _safe_name(self, name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
