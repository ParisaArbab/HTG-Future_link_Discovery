# Architecture

The project represents academic data as a heterogeneous graph:

- **Author nodes**
- **Paper nodes**
- **Concept nodes**
- **Author → writes → Paper edges**
- **Paper → has concept → Concept edges**

The HGT encoder learns one embedding space while keeping node and relationship types separate. A dot-product decoder scores possible paper-concept pairs. High-scoring pairs are returned as predicted missing semantic links.
