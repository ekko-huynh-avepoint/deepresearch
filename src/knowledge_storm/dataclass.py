import dspy
import numpy as np
import re
import threading
from typing import Set, Dict, List, Optional, Union, Tuple

from .encoder import get_encoder
from .interface import Information


class ConversationTurn:
    """Represents a single conversational exchange."""

    def __init__(
        self,
        role: str,
        raw_utterance: str,
        utterance_type: str,
        claim_to_make: Optional[str] = None,
        utterance: Optional[str] = None,
        queries: Optional[List[str]] = None,
        raw_retrieved_info: Optional[List[Information]] = None,
        cited_info: Optional[Dict[int, Information]] = None,
    ):
        self.utterance = utterance or raw_utterance
        self.raw_utterance = raw_utterance
        if ":" in role:
            self.role, self.role_description = [i.strip() for i in role.split(":", 1)]
        else:
            self.role = role
            self.role_description = ""
        self.queries = queries or []
        self.raw_retrieved_info = raw_retrieved_info or []
        self.cited_info = cited_info or {}
        self.utterance_type = utterance_type
        self.claim_to_make = claim_to_make or ""

    def get_all_citation_index(self) -> List[int]:
        return list(map(int, re.findall(r"\[(\d+)]", self.utterance)))

    def to_dict(self) -> Dict:
        return {
            "utterance": self.utterance,
            "raw_utterance": self.raw_utterance,
            "role": self.role,
            "role_description": self.role_description,
            "queries": self.queries,
            "utterance_type": self.utterance_type,
            "claim_to_make": self.claim_to_make,
            "raw_retrieved_info": [info.to_dict() for info in self.raw_retrieved_info],
            "cited_info": None,  # Not serializing cited_info as it's rebuilt on load
        }

    @classmethod
    def from_dict(cls, data: Dict):
        raw_retrieved_info = [
            Information.from_dict(info) for info in data.get("raw_retrieved_info", [])
        ]
        return cls(
            utterance=data.get("utterance"),
            raw_utterance=data.get("raw_utterance"),
            role=f"{data.get('role', '')}: {data.get('role_description', '')}".strip(": "),
            queries=data.get("queries", []),
            raw_retrieved_info=raw_retrieved_info,
            cited_info=None,
            utterance_type=data.get("utterance_type"),
            claim_to_make=data.get("claim_to_make"),
        )


class KnowledgeNode:
    """A node in the hierarchical knowledge base."""

    def __init__(
        self,
        name: str,
        content: Optional[Set[int]] = None,
        parent: Optional["KnowledgeNode"] = None,
        children: Optional[List["KnowledgeNode"]] = None,
        synthesize_output: Optional[str] = None,
        need_regenerate_synthesize_output: bool = True,
    ):
        self.name = name
        self.content: Set[int] = set(content or [])
        self.children: List[KnowledgeNode] = children or []
        self.parent = parent
        self.synthesize_output = synthesize_output
        self.need_regenerate_synthesize_output = need_regenerate_synthesize_output

    def has_child(self, child_node_name: str) -> bool:
        return any(child.name == child_node_name for child in self.children)

    def add_child(self, child_node_name: str, duplicate_handling: str = "skip") -> "KnowledgeNode":
        if self.has_child(child_node_name):
            if duplicate_handling == "skip":
                return next(child for child in self.children if child.name == child_node_name)
            if duplicate_handling == "raise error":
                raise ValueError(f"Node {child_node_name} already exists under {self.name}.")
        child = KnowledgeNode(name=child_node_name, parent=self)
        self.children.append(child)
        return child

    def collect_all_content(self) -> Set[int]:
        """Recursively collects all content from self and descendants."""
        all_content = set(self.content)
        for child in self.children:
            all_content.update(child.collect_all_content())
        return all_content

    def get_parent(self) -> Optional["KnowledgeNode"]:
        return self.parent

    def get_children(self) -> List["KnowledgeNode"]:
        return self.children

    def get_children_names(self) -> List[str]:
        return [child.name for child in self.children]

    def get_path_from_root(self, root: Optional["KnowledgeNode"] = None) -> List[str]:
        path = []
        current = self
        while current:
            path.append(current.name)
            if root and current.name == root.name:
                break
            current = current.parent
        return list(reversed(path))

    def insert_information(self, info_idx: int):
        if info_idx not in self.content:
            self.need_regenerate_synthesize_output = True
            self.content.add(info_idx)

    def get_all_descendents(self) -> List["KnowledgeNode"]:
        desc = []
        for child in self.children:
            desc.append(child)
            desc.extend(child.get_all_descendents())
        return desc

    def get_all_predecessors(self) -> List["KnowledgeNode"]:
        preds = []
        curr = self.parent
        while curr:
            preds.append(curr)
            curr = curr.parent
        return preds

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "content": list(self.content),
            "children": [child.to_dict() for child in self.children],
            "parent": self.parent.name if self.parent else None,
            "synthesize_output": self.synthesize_output,
            "need_regenerate_synthesize_output": self.need_regenerate_synthesize_output,
        }

    @classmethod
    def from_dict(cls, data: Dict, parent_node: Optional["KnowledgeNode"] = None) -> "KnowledgeNode":
        node = cls(
            name=data["name"],
            content=set(data.get("content", [])),
            parent=parent_node,
            synthesize_output=data.get("synthesize_output"),
            need_regenerate_synthesize_output=data.get("need_regenerate_synthesize_output", True),
        )
        node.children = [cls.from_dict(child_data, parent_node=node) for child_data in data.get("children", [])]
        return node

    def __repr__(self):
        return f"KnowledgeNode(name={self.name}, content={self.content}, children={len(self.children)})"


class KnowledgeBase:
    """
    Dynamic, hierarchical mind map for collaborative discourse and knowledge tracking.
    """

    def __init__(
        self,
        topic: str,
        knowledge_base_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        node_expansion_trigger_count: int,
        encoder: Optional[object] = None,
    ):
        from .collaborative_storm.modules.article_generation import ArticleGenerationModule
        from .collaborative_storm.modules.information_insertion_module import InsertInformationModule, ExpandNodeModule
        from .collaborative_storm.modules.knowledge_base_summary import KnowledgeBaseSummaryModule

        self.topic = topic
        # Use the singleton encoder unless explicitly provided for testing/mocking
        self.encoder = encoder if encoder is not None else get_encoder()

        self.information_insert_module = InsertInformationModule(engine=knowledge_base_lm, encoder=self.encoder)
        self.expand_node_module = ExpandNodeModule(
            engine=knowledge_base_lm,
            information_insert_module=self.information_insert_module,
            node_expansion_trigger_count=node_expansion_trigger_count,
        )
        self.article_generation_module = ArticleGenerationModule(engine=knowledge_base_lm)
        self.gen_summary_module = KnowledgeBaseSummaryModule(engine=knowledge_base_lm)

        self.root = KnowledgeNode(name="root")
        self.kb_embedding = {
            "hash": hash(""),
            "encoded_structure": np.array([[]]),
            "structure_string": "",
        }
        self.info_uuid_to_info_dict: Dict[int, Information] = {}
        self.info_hash_to_uuid_dict: Dict[int, int] = {}
        self._lock = threading.Lock()

    def to_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "tree": self.root.to_dict(),
            "info_uuid_to_info_dict": {k: v.to_dict() for k, v in self.info_uuid_to_info_dict.items()},
            "info_hash_to_uuid_dict": self.info_hash_to_uuid_dict,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        knowledge_base_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        node_expansion_trigger_count: int,
        encoder: Optional[object] = None,
    ):
        kb = cls(
            topic=data["topic"],
            knowledge_base_lm=knowledge_base_lm,
            node_expansion_trigger_count=node_expansion_trigger_count,
            encoder=encoder if encoder is not None else get_encoder(),
        )
        kb.root = KnowledgeNode.from_dict(data["tree"])
        kb.info_hash_to_uuid_dict = {int(k): int(v) for k, v in data["info_hash_to_uuid_dict"].items()}
        kb.info_uuid_to_info_dict = {int(k): Information.from_dict(v) for k, v in data["info_uuid_to_info_dict"].items()}
        return kb

    def get_knowledge_base_structure_embedding(
        self, root: Optional[KnowledgeNode] = None
    ) -> Tuple[np.ndarray, List[str]]:
        outline_str = self.get_node_hierarchy_string(
            include_indent=False, include_full_path=True, include_hash_tag=False, root=root
        )
        outline_str_hash = hash(outline_str)
        if outline_str_hash != self.kb_embedding["hash"]:
            outline_lines = outline_str.split("\n")
            cleaned = [line.replace(" -> ", ", ") for line in outline_lines]
            encoded = self.encoder.encode(cleaned)
            self.kb_embedding = {
                "hash": outline_str_hash,
                "encoded_structure": encoded,
                "structure_string": outline_lines,
            }
        return self.kb_embedding["encoded_structure"], self.kb_embedding["structure_string"]

    def traverse_down(self, node: KnowledgeNode) -> List[KnowledgeNode]:
        nodes = []
        def _traverse(cur):
            nodes.append(cur)
            for child in cur.get_children():
                _traverse(child)
        _traverse(node)
        return nodes

    def traverse_up(self, node: KnowledgeNode) -> List[KnowledgeNode]:
        nodes = []
        while node:
            nodes.append(node)
            node = node.get_parent()
        return nodes

    def collect_all_nodes(self) -> List[KnowledgeNode]:
        nodes = []
        def _collect(node):
            nodes.append(node)
            for child in node.children:
                _collect(child)
        _collect(self.root)
        return nodes

    def insert_node(
        self,
        new_node_name: str,
        parent_node: Optional[KnowledgeNode] = None,
        duplicate_handling: str = "skip",
    ) -> KnowledgeNode:
        if parent_node is None:
            return self.root.add_child(new_node_name, duplicate_handling=duplicate_handling)
        return parent_node.add_child(new_node_name, duplicate_handling=duplicate_handling)

    def find_node(self, current_node: KnowledgeNode, node_name: str) -> Optional[KnowledgeNode]:
        if current_node.name == node_name:
            return current_node
        for child in current_node.get_children():
            result = self.find_node(child, node_name)
            if result:
                return result
        return None

    def insert_from_outline_string(self, outline_string: str, duplicate_handling: str = "skip"):
        last_node_at_level = {}
        for line in outline_string.split("\n"):
            level = line.count("#")
            if level > 0:
                title = line.strip("# ").strip()
                if title.lower() in ["overview", "summary", "introduction"]:
                    continue
                parent_node = None if level == 1 else last_node_at_level.get(level - 1)
                new_node = self.insert_node(
                    new_node_name=title,
                    parent_node=parent_node,
                    duplicate_handling=duplicate_handling,
                )
                last_node_at_level[level] = new_node
                for deeper_level in list(last_node_at_level.keys()):
                    if deeper_level > level:
                        del last_node_at_level[deeper_level]

    def get_node_hierarchy_string(
        self,
        include_indent: bool = False,
        include_full_path: bool = False,
        include_hash_tag: bool = True,
        include_node_content_count: bool = False,
        cited_indices: Optional[List[int]] = None,
        root: Optional[KnowledgeNode] = None,
    ) -> str:
        def find_node_contain_index(node, index):
            found = []
            def _traverse(cur):
                if cur and index in cur.content:
                    found.append(cur)
                for child in cur.get_children():
                    _traverse(child)
            _traverse(node)
            return found

        paths_to_highlight = set()
        nodes_to_include = set()
        if cited_indices:
            for idx in cited_indices:
                for cur_node in find_node_contain_index(self.root, idx):
                    paths_to_highlight.add(" -> ".join(cur_node.get_path_from_root()))
                    nodes_to_include.add(cur_node)
                    nodes_to_include.update(cur_node.get_all_descendents())
                    preds = cur_node.get_all_predecessors()
                    for pred in preds:
                        nodes_to_include.update(pred.children)
                    nodes_to_include.update(preds)

        def should_include_node(node):
            return True if cited_indices is None else node in nodes_to_include

        def should_omit_child_nodes(node):
            if cited_indices is None:
                return False
            return not any(should_include_node(child) for child in node.children)

        def helper(cur_root, level):
            out = []
            if cur_root:
                if should_include_node(cur_root):
                    indent = "" if not include_indent else "\t" * (level - 1)
                    full_path = " -> ".join(cur_root.get_path_from_root(root=root))
                    node_info = cur_root.name if not include_full_path else full_path
                    hash_tag = "#" * level + " " if include_hash_tag else ""
                    count = f" ({len(cur_root.content)})" if include_node_content_count else ""
                    special = "" if not (cited_indices and full_path in paths_to_highlight) else " ⭐"
                    out.append(f"{indent}{hash_tag}{node_info}{count}{special}")
                    if should_omit_child_nodes(cur_root):
                        if cur_root.children:
                            child_indent = "" if not include_indent else "\t" * level
                            out.append(f"{child_indent}...")
                    else:
                        for child in cur_root.children:
                            out.extend(helper(child, level + 1))
            return out

        result = []
        if root is None and self.root:
            for child in self.root.children:
                result.extend(helper(child, 1))
        else:
            result.extend(helper(root, 1))
        return "\n".join(result)

    def find_node_by_path(
        self,
        path: str,
        missing_node_handling: str = "abort",
        root: Optional[KnowledgeNode] = None,
    ) -> Optional[KnowledgeNode]:
        node_names = path.split(" -> ")
        current = self.root if root is None else root
        for name in node_names[1:]:
            found = next((child for child in current.children if child.name == name), None)
            if found is None:
                if missing_node_handling == "abort":
                    return None
                elif missing_node_handling == "create":
                    current = current.add_child(name)
                elif missing_node_handling == "raise error":
                    structure = self.get_node_hierarchy_string(
                        include_indent=True, include_full_path=False, include_hash_tag=True
                    )
                    raise ValueError(f"Cannot find node {{{name}}} under {{{current.name}}}\n{structure}")
            else:
                current = found
        return current

    def insert_information(
        self,
        path: str,
        information: Information,
        missing_node_handling: str = "abort",
        root: Optional[KnowledgeNode] = None,
    ):
        with self._lock:
            target_node = self.find_node_by_path(path, missing_node_handling, root)
            info_hash = hash(information)
            if information.citation_uuid == -1:
                info_citation_uuid = self.info_hash_to_uuid_dict.get(
                    info_hash, len(self.info_hash_to_uuid_dict) + 1
                )
                information.citation_uuid = info_citation_uuid
                self.info_hash_to_uuid_dict[info_hash] = info_citation_uuid
                self.info_uuid_to_info_dict[info_citation_uuid] = information
            if target_node:
                self.info_uuid_to_info_dict[information.citation_uuid].meta["placement"] = (
                    " -> ".join(target_node.get_path_from_root())
                )
                target_node.insert_information(information.citation_uuid)

    def trim_empty_leaf_nodes(self):
        def trim(node):
            if not node.children and not node.content:
                return True
            node.children = [child for child in node.children if not trim(child)]
            return not node.children and not node.content

        while True:
            before = len(self.get_all_leaf_nodes())
            trim(self.root)
            after = len(self.get_all_leaf_nodes())
            if before == after:
                break

    def get_all_leaf_nodes(self) -> List[KnowledgeNode]:
        leaf_nodes = []
        def find_leaves(node):
            if not node.children:
                leaf_nodes.append(node)
            for child in node.children:
                find_leaves(child)
        find_leaves(self.root)
        return leaf_nodes

    def merge_single_child_nodes(self):
        def merge(node):
            for child in node.children:
                merge(child)
            if len(node.children) == 1:
                single_child = node.children[0]
                node.content.update(single_child.content)
                node.children = single_child.children
                for grandchild in node.children:
                    grandchild.parent = node
        merge(self.root)

    def update_all_info_path(self):
        def _helper(node):
            for citation_idx in node.content:
                self.info_uuid_to_info_dict[citation_idx].meta["placement"] = (
                    " -> ".join(node.get_path_from_root())
                )
            for child in node.children:
                _helper(child)
        _helper(self.root)

    def update_from_conv_turn(
        self,
        conv_turn: ConversationTurn,
        allow_create_new_node: bool = False,
        insert_under_root: bool = False,
    ):
        if not conv_turn:
            return
        info_to_insert = list(conv_turn.cited_info.values())
        if insert_under_root:
            for info in info_to_insert:
                self.insert_information(path=self.root.name, information=info)
        else:
            self.information_insert_module(
                knowledge_base=self,
                information=info_to_insert,
                allow_create_new_node=allow_create_new_node,
            )
        old_to_new_mapping = {old: info.citation_uuid for old, info in conv_turn.cited_info.items()}
        for old_idx, new_idx in old_to_new_mapping.items():
            conv_turn.utterance = conv_turn.utterance.replace(f"[{old_idx}]", f"[_{new_idx}_]")
            conv_turn.raw_utterance = conv_turn.raw_utterance.replace(f"[{old_idx}]", f"[_{new_idx}_]")
        for _, new_idx in old_to_new_mapping.items():
            conv_turn.utterance = conv_turn.utterance.replace(f"[_{new_idx}_]", f"[{new_idx}]").replace("[-1]", "")
            conv_turn.raw_utterance = conv_turn.raw_utterance.replace(f"[_{new_idx}_]", f"[{new_idx}]").replace("[-1]", "")
        conv_turn.cited_info = None

    def get_knowledge_base_summary(self) -> str:
        return self.gen_summary_module(self)

    def reogranize(self):
        self.trim_empty_leaf_nodes()
        self.merge_single_child_nodes()
        self.expand_node_module(knowledge_base=self)
        self.trim_empty_leaf_nodes()
        self.merge_single_child_nodes()
        self.update_all_info_path()

    def to_report(self) -> str:
        return self.article_generation_module(self)