import pandas as pd
from enum import Enum
from itertools import chain
from typing import Optional
from biocypher._logger import logger

logger.debug(f"Loading module {__name__}.")

class CompoundWikiAdapterNodeType(Enum):
    """
    Define types of nodes the adapter can provide.
    """
    CHEMICAL = ":Chemical"
    WEBPAGE = ":WebPage"

class CompoundWikiAdapterChemicalField(Enum):
    """
    Define possible fields the adapter can provide for chemicals.
    """
    NAME = "ChemicalName"
    CAS = "ChemicalCAS"
    SMILES = "SMILES"                   # New property
    INCHIKEY = "InChIKey"               # New property

class CompoundWikiAdapterEdgeType(Enum):
    """
    Define possible edges the adapter can provide.
    """
    chemical_webpage = "chemical_webpage"

class CompoundWikiAdapter:
    """
    Adapter for creating a knowledge graph
    """

    def __init__(
        self,
        node_types: Optional[list] = None,
        node_fields: Optional[list] = None,
        edge_types: Optional[list] = None,
        edge_fields: Optional[list] = None,
    ):
        self._set_types_and_fields(node_types, node_fields, edge_types, edge_fields)

    def get_nodes(self):
        """
        Returns a generator of node tuples for node types specified in the
        adapter constructor.
        """

        logger.info("Generating nodes.")

        node_count = 0
        data = pd.read_csv("data/CompoundWiki.csv", dtype=str)
        data = data.map(lambda x: x.replace("'", "") if isinstance(x, str) else x) #FIXME
        for index, row in data.iterrows():
            _id = row["id"]
            _type = row["labels"]

            if _type not in self.node_types:
                logger.info(f"Skipping node with ID={_id} due to type mismatch.")
                continue

            _props = {}
            if _type == ':Chemical':
                _props['name'] = row.get('ChemicalName', None)
                _props['CAS'] = row.get('ChemicalCAS', None)
                _props['SMILES'] = row.get('SMILES', None)
                _props['InChIKey'] = row.get('InChIKey', None)

            #logger.info(f"Yielding node: ID={_id}, Type={_type}, Properties={_props}")
            node_count += 1
            yield (_id, _type, _props)

        data = pd.read_csv("data/CompoundWiki_webpages.csv", dtype=str)
        for index, row in data.iterrows():
            _id = row["id"]
            _type = row["labels"]

            if _type not in self.node_types:
                logger.info(f"Skipping node with ID={_id} due to type mismatch.")
                continue

            _props = {}
            if _type == ':WebPage':
                _props['URL'] = _id

            #logger.info(f"Yielding node: ID={_id}, Type={_type}, Properties={_props}")
            node_count += 1
            yield (_id, _type, _props)

        logger.info(f"Total nodes generated: {node_count}")

    def get_edges(self):
        """
        Returns a generator of edge tuples for edge types specified in the
        adapter constructor.
        """
        logger.info("Generating edges.")

        edge_count = 0
        data = pd.read_csv("data/CompoundWiki_edges.csv", dtype=str)
        for index, row in data.iterrows():
            if row["type"] not in self.edge_types:
                logger.warning(f"Edge type {row['type']} not in specified edge types.")
                continue

            _id = None  # Edges don't necessarily need unique IDs
            _start = row["start"]
            _end = row["end"]
            _type = row["type"]
            _props = {}
            logger.info(f"Yielding edge: Start={_start}, End={_end}, Type={_type}, Properties={_props}")
            edge_count += 1
            yield (_id, _start, _end, _type, _props)

        logger.info(f"Total edges generated: {edge_count}")

    def _set_types_and_fields(self, node_types, node_fields, edge_types, edge_fields):
        """
        Set the types and fields for nodes and edges, if specified. Otherwise, use defaults.
        """
        if node_types:
            self.node_types = [type.value for type in node_types]
        else:
            self.node_types = [type.value for type in CompoundWikiAdapterNodeType]

        if node_fields:
            self.node_fields = [field.value for field in node_fields]
        else:
            self.node_fields = [
                field.value
                for field in chain(
                    CompoundWikiAdapterChemicalField,
                )
            ]

        if edge_types:
            self.edge_types = [type.value for type in edge_types]
        else:
            self.edge_types = [type.value for type in CompoundWikiAdapterEdgeType]

        if edge_fields:
            self.edge_fields = [field.value for field in edge_fields]
        else:
            self.edge_fields = []
