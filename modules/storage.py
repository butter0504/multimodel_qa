from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


class StorageLayer:
    def __init__(self, cfg=None):
        self.cfg = cfg
        if cfg:
            self._backend = cfg.get("storage.backend", "sqlite")
            self._db_path = cfg.get("storage.sqlite_path", "./data/multimodel_qa.db")
            self._json_path = cfg.get("storage.local_json_path", "./data/local_storage.json")
            self._reports_dir = cfg.get("storage.reports_dir", "./reports/")
        else:
            self._backend = "sqlite"
            self._db_path = "./data/multimodel_qa.db"
            self._json_path = "./data/local_storage.json"
            self._reports_dir = "./reports/"

        self._conn: Optional[sqlite3.Connection] = None
        self._init_storage()

    def _init_storage(self):
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._json_path), exist_ok=True)
        os.makedirs(self._reports_dir, exist_ok=True)

        if self._backend == "sqlite":
            try:
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._create_tables()
            except Exception:
                self._backend = "json"
                self._ensure_json_storage()
        else:
            self._ensure_json_storage()

    def _create_tables(self):
        if self._conn is None:
            return

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                file_name TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                upload_time TEXT NOT NULL,
                basic_info TEXT
            );

            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                result TEXT NOT NULL,
                analysis_time TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS data_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                format_name TEXT,
                sample_count INTEGER DEFAULT 0,
                num_classes INTEGER DEFAULT 0,
                task_type TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS flagged_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_object_id INTEGER NOT NULL,
                sample_id TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (data_object_id) REFERENCES data_objects(id) ON DELETE CASCADE
            );
        """)
        self._conn.commit()

    def _ensure_json_storage(self):
        if not os.path.exists(self._json_path):
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump({"projects": [], "analysis_results": [], "data_objects": [], "flagged_samples": []}, f, ensure_ascii=False, indent=2)

    def add_project(self, project: Dict[str, Any]) -> Optional[int]:
        now = datetime.now().isoformat()

        if self._backend == "sqlite" and self._conn:
            try:
                cursor = self._conn.execute(
                    "INSERT INTO projects (name, file_name, analysis_type, upload_time, basic_info) VALUES (?, ?, ?, ?, ?)",
                    (project["name"], project["file_name"], project["analysis_type"], now, json.dumps(project.get("basic_info", {}), ensure_ascii=False)),
                )
                self._conn.commit()
                return cursor.lastrowid
            except Exception:
                pass

        return self._add_project_json(project, now)

    def _add_project_json(self, project: Dict[str, Any], timestamp: str) -> Optional[int]:
        try:
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            project_id = len(data["projects"]) + 1
            data["projects"].append({
                "id": project_id,
                "name": project["name"],
                "file_name": project["file_name"],
                "analysis_type": project["analysis_type"],
                "upload_time": timestamp,
                "basic_info": project.get("basic_info", {}),
            })
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return project_id
        except Exception:
            return None

    def add_analysis_result(self, project_id: int, result: Dict[str, Any]) -> Optional[int]:
        now = datetime.now().isoformat()

        if self._backend == "sqlite" and self._conn:
            try:
                cursor = self._conn.execute(
                    "INSERT INTO analysis_results (project_id, result, analysis_time) VALUES (?, ?, ?)",
                    (project_id, json.dumps(result, ensure_ascii=False, default=str), now),
                )
                self._conn.commit()
                return cursor.lastrowid
            except Exception:
                pass

        return self._add_result_json(project_id, result, now)

    def _add_result_json(self, project_id: int, result: Dict[str, Any], timestamp: str) -> Optional[int]:
        try:
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            result_id = len(data["analysis_results"]) + 1
            data["analysis_results"].append({
                "id": result_id,
                "project_id": project_id,
                "result": result,
                "analysis_time": timestamp,
            })
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return result_id
        except Exception:
            return None

    def add_data_object(self, project_id: int, data_obj_info: Dict[str, Any]) -> Optional[int]:
        now = datetime.now().isoformat()

        if self._backend == "sqlite" and self._conn:
            try:
                cursor = self._conn.execute(
                    "INSERT INTO data_objects (project_id, format_name, sample_count, num_classes, task_type, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        project_id,
                        data_obj_info.get("source_format", "unknown"),
                        data_obj_info.get("sample_count", 0),
                        data_obj_info.get("num_classes", 0),
                        data_obj_info.get("task_type", "unknown"),
                        json.dumps(data_obj_info.get("metadata", {}), ensure_ascii=False),
                        now,
                    ),
                )
                self._conn.commit()
                return cursor.lastrowid
            except Exception:
                pass

        return None

    def add_flagged_sample(self, data_object_id: int, sample_id: str, reason: str = "") -> Optional[int]:
        now = datetime.now().isoformat()

        if self._backend == "sqlite" and self._conn:
            try:
                cursor = self._conn.execute(
                    "INSERT INTO flagged_samples (data_object_id, sample_id, reason, created_at) VALUES (?, ?, ?, ?)",
                    (data_object_id, sample_id, reason, now),
                )
                self._conn.commit()
                return cursor.lastrowid
            except Exception:
                pass

        return None

    def get_projects(self, limit: int = 10) -> List[Dict[str, Any]]:
        if self._backend == "sqlite" and self._conn:
            try:
                cursor = self._conn.execute(
                    "SELECT id, name, file_name, analysis_type, upload_time, basic_info FROM projects ORDER BY upload_time DESC LIMIT ?",
                    (limit,),
                )
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "id": row[0],
                        "name": row[1],
                        "file_name": row[2],
                        "analysis_type": row[3],
                        "upload_time": row[4],
                        "basic_info": json.loads(row[5]) if row[5] else {},
                    })
                return results
            except Exception:
                pass

        return self._get_projects_json(limit)

    def _get_projects_json(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            projects = sorted(data.get("projects", []), key=lambda x: x.get("upload_time", ""), reverse=True)
            return projects[:limit]
        except Exception:
            return []

    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        if self._backend == "sqlite" and self._conn:
            try:
                cursor = self._conn.execute(
                    "SELECT id, name, file_name, analysis_type, upload_time, basic_info FROM projects WHERE id = ?",
                    (project_id,),
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "name": row[1],
                        "file_name": row[2],
                        "analysis_type": row[3],
                        "upload_time": row[4],
                        "basic_info": json.loads(row[5]) if row[5] else {},
                    }
            except Exception:
                pass

        return self._get_project_json(project_id)

    def _get_project_json(self, project_id: int) -> Optional[Dict[str, Any]]:
        try:
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for p in data.get("projects", []):
                if p.get("id") == project_id:
                    return p
        except Exception:
            pass
        return None

    def get_analysis_result(self, project_id: int) -> Optional[Dict[str, Any]]:
        if self._backend == "sqlite" and self._conn:
            try:
                cursor = self._conn.execute(
                    "SELECT result FROM analysis_results WHERE project_id = ? ORDER BY analysis_time DESC LIMIT 1",
                    (project_id,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
            except Exception:
                pass

        return self._get_result_json(project_id)

    def _get_result_json(self, project_id: int) -> Optional[Dict[str, Any]]:
        try:
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = [r for r in data.get("analysis_results", []) if r.get("project_id") == project_id]
            if results:
                results.sort(key=lambda x: x.get("analysis_time", ""), reverse=True)
                return results[0].get("result")
        except Exception:
            pass
        return None

    def save_report(self, content: str, filename: Optional[str] = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.json"

        filepath = os.path.join(self._reports_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
