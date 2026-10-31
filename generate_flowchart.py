"""
Generates clean Excalidraw-styled architecture diagrams for README.md:
1. architecture_flowchart.png (Full Dual-Engine Architecture with no overlapping titles)
2. architecture_pipeline.png (Linear Flowchart matching the second picture)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Common Excalidraw styling constants
EXCALI_BG = "#FFFFFF"         # Crisp clean canvas
TEXT_COLOR = "#1E1E1E"        # Charcoal ink
SUBTEXT_COLOR = "#4A5568"     # Soft slate
BORDER_COLOR = "#2D3748"      # Hand-drawn dark slate border
ARROW_COLOR = "#4A5568"

# Excalidraw muted, hand-drawn pastel palette
BOX_BLUE = "#EBF8FF"          # Light cyan-blue
BORDER_BLUE = "#3182CE"

BOX_PURPLE = "#FAF5FF"        # Light violet
BORDER_PURPLE = "#805AD5"

BOX_GREEN = "#F0FDF4"         # Light sage green
BORDER_GREEN = "#38A169"

BOX_AMBER = "#FFFBEB"         # Light warm amber
BORDER_AMBER = "#D97706"

BOX_GRAY = "#F8FAFC"          # Light soft gray
BORDER_GRAY = "#4A5568"


# ==============================================================================
# DIAGRAM 1: Full Architecture (Fixed overlaps & Excalidraw Styling)
# ==============================================================================
def create_full_architecture():
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300)
    fig.patch.set_facecolor(EXCALI_BG)
    ax.set_facecolor(EXCALI_BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    # Header Titles - Placed high at y=9.0 with plenty of breathing room
    ax.text(8, 9.0, "Dual-Engine Conversational Data Analytics Assistant", 
            ha='center', va='center', fontsize=20, fontweight='bold', color=TEXT_COLOR, fontfamily='sans-serif')
    ax.text(8, 8.55, "Zero-Hallucination Architecture: Semantic RAG + Deterministic Pandas Execution with Strict AST Trust Boundary", 
            ha='center', va='center', fontsize=11, color=SUBTEXT_COLOR, fontfamily='sans-serif')

    def draw_box(x, y, w, h, title, subtitle, bg_col, border_col, badge_text=None):
        # Main box with subtle rounded corners
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                     facecolor=bg_col, edgecolor=border_col, linewidth=1.8)
        ax.add_patch(rect)
        
        # Subtle badge header inside the box
        if badge_text:
            b_w = len(badge_text) * 0.11 + 0.25
            b_rect = patches.FancyBboxPatch((x + 0.15, y + h - 0.32), b_w, 0.22,
                                           boxstyle="round,pad=0.02,rounding_size=0.06",
                                           facecolor=border_col, edgecolor='none')
            ax.add_patch(b_rect)
            ax.text(x + 0.15 + b_w/2, y + h - 0.21, badge_text,
                    ha='center', va='center', fontsize=7.5, fontweight='bold', color="#FFFFFF")

        text_y = y + h/2 + 0.08 if subtitle else y + h/2
        ax.text(x + w/2, text_y, title, ha='center', va='center', 
                fontsize=11, fontweight='bold', color=TEXT_COLOR, fontfamily='sans-serif')
        if subtitle:
            ax.text(x + w/2, y + h/2 - 0.22, subtitle, ha='center', va='center', 
                    fontsize=8.5, color=SUBTEXT_COLOR, fontfamily='sans-serif')

    def draw_arrow(x1, y1, x2, y2, label=None, color=ARROW_COLOR, style='->', lw=1.6, label_offset=(0, 0.18)):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=5, shrinkB=5))
        if label:
            mid_x = (x1 + x2) / 2 + label_offset[0]
            mid_y = (y1 + y2) / 2 + label_offset[1]
            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8, 
                    color=color, fontweight='bold', backgroundcolor=EXCALI_BG)

    # 1. User Ingestion (Left Center)
    draw_box(0.6, 4.8, 2.5, 1.4, "User Query", "NL Prompt + Raw Dataset", BOX_GRAY, BORDER_GRAY, "INGESTION")

    # 2. Semantic Query Router (Center Left)
    draw_box(3.8, 4.8, 2.7, 1.4, "Semantic Query Router", "Intent Classifier (NLP)", BOX_BLUE, BORDER_BLUE, "TRIAGE")
    draw_arrow(3.1, 5.5, 3.8, 5.5, color=BORDER_BLUE)

    # 3. Local RAG Vector Engine (Top Center) - y=6.6 to 8.0, cleanly below title (y=8.55)
    draw_arrow(5.15, 6.2, 5.15, 7.3, color=BORDER_PURPLE)
    draw_arrow(5.15, 7.3, 7.5, 7.3, "KNOWLEDGE", color=BORDER_PURPLE, label_offset=(0, 0.18))
    draw_box(7.5, 6.6, 3.2, 1.4, "Local RAG Vector Engine", "TF-IDF + Cosine Similarity", BOX_PURPLE, BORDER_PURPLE, "SEMANTIC RAG")

    # 4. LLM Analysis Planner (Center)
    draw_arrow(6.5, 5.5, 7.5, 5.5, "ANALYTICAL", color=BORDER_BLUE, label_offset=(0, 0.18))
    draw_box(7.5, 4.8, 3.2, 1.4, "LLM Analysis Planner", "AnalysisPlan Schema (JSON)", BOX_BLUE, BORDER_BLUE, "PLANNING")

    # 5. Dataset Profiler (Bottom Left)
    draw_box(3.8, 2.3, 2.7, 1.4, "Dataset Profiler", "Schema, Types & Stats", BOX_GRAY, BORDER_GRAY, "OBSERVATION")
    draw_arrow(1.85, 4.8, 1.85, 3.0, color=ARROW_COLOR)
    draw_arrow(1.85, 3.0, 3.8, 3.0, "Active DF", color=ARROW_COLOR)
    draw_arrow(5.15, 3.7, 5.15, 4.8, "Context", color=ARROW_COLOR)
    draw_arrow(6.5, 3.0, 7.5, 3.0, color=BORDER_GREEN)

    # 6. Trust Boundary Box (Enclosing Validator & Deterministic Execution)
    tb_rect = patches.FancyBboxPatch((7.2, 0.5), 7.9, 3.7, boxstyle="round,pad=0.12,rounding_size=0.18",
                                    facecolor="#F7FEE7", edgecolor=BORDER_GREEN, linewidth=1.5, linestyle='--')
    ax.add_patch(tb_rect)
    ax.text(7.45, 3.9, "STRICT AST TRUST BOUNDARY & NUMERICAL INTEGRITY GUARANTEE", 
            fontsize=8.5, fontweight='bold', color=BORDER_GREEN, fontfamily='sans-serif')

    # Strict Plan Validator
    draw_box(7.5, 2.2, 3.2, 1.3, "Strict Plan Validator", "Column, Type & Op Verification", BOX_GREEN, BORDER_GREEN, "VALIDATOR")
    draw_arrow(9.1, 4.8, 9.1, 3.5, "AnalysisPlan", color=BORDER_BLUE)

    # Deterministic Pandas Engine
    draw_box(11.5, 2.2, 3.2, 1.3, "Pandas Calculation Engine", "Deterministic Aggregations", BOX_GREEN, BORDER_GREEN, "AUTHORITY")
    draw_arrow(10.7, 2.85, 11.5, 2.85, "Approved IR", color=BORDER_GREEN)

    # RAG Grounded Answer connection to Delivery
    draw_arrow(10.7, 7.3, 13.1, 7.3, color=BORDER_PURPLE)
    draw_arrow(13.1, 7.3, 13.1, 6.2, "Grounded Text", color=BORDER_PURPLE)

    # Analytical Pandas connection to Delivery
    draw_arrow(13.1, 3.5, 13.1, 4.8, "Exact Values", color=BORDER_GREEN)

    # Output: 4-Part Response
    draw_box(11.5, 4.8, 3.2, 1.4, "4-Part UI Response", "Answer + Table + Chart + Audit", BOX_AMBER, BORDER_AMBER, "DELIVERY")

    plt.tight_layout()
    plt.savefig("architecture_flowchart.png", dpi=300, bbox_inches='tight', facecolor=EXCALI_BG)
    plt.close()
    print("Successfully generated clean Excalidraw architecture_flowchart.png")


# ==============================================================================
# DIAGRAM 2: Linear Pipeline Flowchart (Matching User's 2nd Picture)
# ==============================================================================
def create_pipeline_flowchart():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    fig.patch.set_facecolor(EXCALI_BG)
    ax.set_facecolor(EXCALI_BG)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Header
    ax.text(7.5, 7.9, "Conversational Query Routing & Analysis Execution Pipeline", 
            ha='center', va='center', fontsize=18, fontweight='bold', color=TEXT_COLOR, fontfamily='sans-serif')
    ax.text(7.5, 7.45, "End-to-End Request Flow: Triage -> Knowledge Retrieval -> Plan Validation -> Deterministic Pandas Execution", 
            ha='center', va='center', fontsize=10.5, color=SUBTEXT_COLOR, fontfamily='sans-serif')

    def draw_box(x, y, w, h, title, subtitle=None, bg_col=BOX_GRAY, border_col=BORDER_GRAY):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                     facecolor=bg_col, edgecolor=border_col, linewidth=1.6)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h/2 + 0.12, title, ha='center', va='center', 
                    fontsize=10.5, fontweight='bold', color=TEXT_COLOR, fontfamily='sans-serif')
            ax.text(x + w/2, y + h/2 - 0.15, subtitle, ha='center', va='center', 
                    fontsize=8.5, color=SUBTEXT_COLOR, fontfamily='sans-serif')
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', 
                    fontsize=10.5, fontweight='bold', color=TEXT_COLOR, fontfamily='sans-serif')

    def draw_arrow(x1, y1, x2, y2, label=None, color=ARROW_COLOR, style='->', lw=1.6, label_offset=(0, 0.15)):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=5, shrinkB=5))
        if label:
            mid_x = (x1 + x2) / 2 + label_offset[0]
            mid_y = (y1 + y2) / 2 + label_offset[1]
            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8.5, 
                    color=color, fontweight='bold', backgroundcolor=EXCALI_BG)

    # ROW 1 (Top Level): User Query -> Router -> LLM Planner
    # Box 1: User Query & CSV
    draw_box(0.8, 5.6, 3.2, 1.2, "User Query & CSV", "Input Prompt & Data File", BOX_GRAY, BORDER_GRAY)
    draw_arrow(4.0, 6.2, 5.4, 6.2, color=BORDER_BLUE)

    # Box 2: Semantic Query Router
    draw_box(5.4, 5.6, 3.8, 1.2, "Semantic Query Router", "Intent Classifier (NLP)", BOX_BLUE, BORDER_BLUE)
    draw_arrow(9.2, 6.2, 10.6, 6.2, color=BORDER_BLUE)

    # Box 3: LLM Analysis Planner
    draw_box(10.6, 5.6, 3.6, 1.2, "LLM Analysis Planner", "Generates Pydantic Schema", BOX_BLUE, BORDER_BLUE)

    # ROW 2 (Middle Level): RAG Vector Store & Strict Plan Validator
    # Down arrow from Router to Local RAG Vector Store
    draw_arrow(7.3, 5.6, 7.3, 3.8, "Knowledge Queries", color=BORDER_PURPLE, label_offset=(1.2, 0))
    draw_box(5.4, 2.6, 3.8, 1.2, "Local RAG Vector Store", "TF-IDF + Cosine Similarity", BOX_PURPLE, BORDER_PURPLE)

    # Down arrow from LLM Planner to Validator
    draw_arrow(12.4, 5.6, 12.4, 4.2, "Structured AnalysisPlan (JSON)", color=BORDER_BLUE, label_offset=(1.8, 0))

    # Box 4: Strict Plan Validator
    draw_box(10.6, 3.0, 3.6, 1.2, "Strict Plan Validator", "(AST Trust Boundary)", BOX_GREEN, BORDER_GREEN)

    # Down arrow from Validator to Pandas Execution
    draw_arrow(12.4, 3.0, 12.4, 1.8, "Approved Execution Plan", color=BORDER_GREEN, label_offset=(1.6, 0))

    # ROW 3 (Bottom Level): 4-Part Response <- Chart Generator <- Deterministic Pandas Engine
    # Box 5: Deterministic Pandas Engine
    draw_box(10.6, 0.6, 3.6, 1.2, "Deterministic Pandas", "Execution Engine", BOX_GREEN, BORDER_GREEN)
    draw_arrow(10.6, 1.2, 9.2, 1.2, color=BORDER_GREEN)

    # Down arrow from RAG Store to Chart/UI Delivery
    draw_arrow(7.3, 2.6, 7.3, 1.8, color=BORDER_PURPLE)

    # Box 6: Interactive Plotly / UI Generator
    draw_box(5.4, 0.6, 3.8, 1.2, "Interactive Plotly / UI", "Chart Generator & Formatter", BOX_AMBER, BORDER_AMBER)
    draw_arrow(5.4, 1.2, 4.0, 1.2, color=BORDER_AMBER)

    # Box 7: 4-Part Response
    draw_box(0.8, 0.6, 3.2, 1.2, "4-Part Response", "(Answer + Table + Chart + Audit)", BOX_AMBER, BORDER_AMBER)

    plt.tight_layout()
    plt.savefig("architecture_pipeline.png", dpi=300, bbox_inches='tight', facecolor=EXCALI_BG)
    plt.close()
    print("Successfully generated architecture_pipeline.png")


if __name__ == "__main__":
    create_full_architecture()
    create_pipeline_flowchart()
