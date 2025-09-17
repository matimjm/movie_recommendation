from graphviz import Digraph

# Create a Digraph
dot = Digraph(comment="Movie Recommendation Workflow", format="png")
dot.attr(rankdir="LR", size="8,5")

# Step 1: Data Collection
dot.node("1", "Data Collection\n- Download raw movie data\n- Attributes: id, title, overview, genres, revenue, budget, etc.", shape="box")

# Step 2: Data Preparation and Cleaning
dot.node("2", "Data Preparation & Cleaning\n- Load into Pandas\n- Clean text & numerical data\n- Feature engineering\n- Normalize numerical features\n- Final DataFrame: id, text_soup, features", shape="box")

# Step 3: Model Training & Embedding Generation
dot.node("3", "Model Training & Embedding\n- Tokenize text_soup\n- Generate embeddings (BERT)\n- Save embeddings", shape="box")

# Step 4: Recommendation Logic
dot.node("4", "Recommendation Logic\n- User input: movie + preferences\n- Weighted query vector\n- Cosine similarity search\n- Rank & filter results", shape="box")

# Step 5: Final Output
dot.node("5", "Final Output\n- Display recommended movies\n- Retrieve title, poster, overview", shape="box")

# Edges
dot.edges([("1", "2"), ("2", "3"), ("3", "4"), ("4", "5")])

# Save and render
dot.graph_attr.update(dpi="300")  # 300 DPI = print quality
output_path = "workflow_diagram"

dot.render(output_path, format="png", cleanup=True)