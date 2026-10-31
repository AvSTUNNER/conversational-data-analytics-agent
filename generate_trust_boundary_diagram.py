"""
Generates an Excalidraw-styled flowchart for the Strict AST Trust Boundary Dual-Engine Flow:
1. NATURAL LANGUAGE QUERY
2. SEMANTIC QUERY ROUTER (splits into Knowledge Query vs Analytical Query)
3. Branch A: LOCAL RAG VECTOR STORE (TF-IDF + Cosine Sim) -> Grounded Text
4. Branch B: LLM ANALYSIS PLANNER (Pydantic JSON IR) -> STRICT AST TRUST BOUNDARY
   - Rigid AST Schema & Column Validator
   - Deterministic Pandas Execution Engine -> Exact Numbers
5. 4-PART UI RESPONSE (Factual Direct Answer, Formatted Evidence Table, Interactive Plotly Viz, Verifiable Audit Trail)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Excalidraw styling constants
EXCALI_BG = "#FFFFFF"         # Crisp clean white canvas
TEXT_COLOR = "#1E1E1E"        # Charcoal ink
SUBTEXT_COLOR = "#4A5568"     # Soft slate
ARROW_COLOR = "#4A5568"

# Excalidraw hand-drawn pastel palette
BOX_GRAY = "#F8FAFC"
BORDER_GRAY = "#475569"

BOX_BLUE = "#EFF6FF"
BORDER_BLUE = "#3B82F6"

BOX_PURPLE = "#FAF5FF"
BORDER_PURPLE = "#8B5CF6"

BOX_GREEN = "#F0FDF4"
BORDER_GREEN = "#10B981"

BOX_AMBER = "#FFFBEB"
BORDER_AMBER = "#F59E0B"

fig, ax = plt.subplots(figsize=(15, 10), dpi=300)
fig.patch.set_facecolor(EXCALI_BG)
ax.set_facecolor(EXCALI_BG)
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.axis('off')

# Header Title
ax.text(7.5, 9.4, "Dual-Engine Request Routing & AST Trust Boundary Flow", 
        ha='center', va='center', fontsize=18, fontweight='bold', color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(7.5, 9.0, "Decoupled Architecture: Semantic Knowledge RAG + Deterministic Numerical Calculations", 
        ha='center', va='center', fontsize=10.5, color=SUBTEXT_COLOR, fontfamily='sans-serif')

def draw_box(x, y, w, h, title, subtitle=None, bg_col=BOX_GRAY, border_col=BORDER_GRAY):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                 facecolor=bg_col, edgecolor=border_col, linewidth=1.6)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h/2 + 0.12, title, ha='center', va='center', 
                fontsize=10.5, fontweight='bold', color=TEXT_COLOR, fontfamily='sans-serif')
        ax.text(x + w/2, y + h/2 - 0.16, subtitle, ha='center', va='center', 
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

# 1. NATURAL LANGUAGE QUERY (Top Left)
draw_box(1.0, 7.8, 4.4, 0.9, "NATURAL LANGUAGE QUERY", bg_col=BOX_GRAY, border_col=BORDER_GRAY)
draw_arrow(3.2, 7.8, 3.2, 6.9, color=BORDER_BLUE)

# 2. SEMANTIC QUERY ROUTER
draw_box(1.0, 5.9, 4.4, 1.0, "SEMANTIC QUERY ROUTER", bg_col=BOX_BLUE, border_col=BORDER_BLUE)

# Knowledge Query Arrow (Horizontal to Right)
draw_arrow(5.4, 6.4, 8.5, 6.4, "Knowledge Query", color=BORDER_PURPLE, label_offset=(0, 0.22))

# Analytical Query Arrow (Vertical Down)
draw_arrow(3.2, 5.9, 3.2, 4.8, "Analytical Query", color=BORDER_BLUE, label_offset=(1.1, 0))

# 3. LOCAL RAG VECTOR STORE (Top Right)
draw_box(8.5, 5.8, 5.2, 1.2, "LOCAL RAG VECTOR STORE", "(TF-IDF + Cosine Sim)", bg_col=BOX_PURPLE, border_col=BORDER_PURPLE)

# Grounded Text Arrow (Vertical Down from RAG Store to 4-Part UI Response)
draw_arrow(11.1, 5.8, 11.1, 4.4, "Grounded Text", color=BORDER_PURPLE, label_offset=(0.9, 0))

# 4. LLM ANALYSIS PLANNER (Center Left)
draw_box(1.0, 3.8, 4.4, 1.0, "LLM ANALYSIS PLANNER", "(Pydantic JSON IR)", bg_col=BOX_BLUE, border_col=BORDER_BLUE)
draw_arrow(3.2, 3.8, 3.2, 2.7, color=BORDER_GREEN)

# 5. STRICT AST TRUST BOUNDARY BOX (Bottom Left Container)
tb_bg = patches.FancyBboxPatch((0.8, 0.6), 5.6, 2.1, boxstyle="round,pad=0.1,rounding_size=0.18",
                              facecolor="#F7FEE7", edgecolor=BORDER_GREEN, linewidth=1.6, linestyle='--')
ax.add_patch(tb_bg)
ax.text(1.1, 2.45, "[LOCK] STRICT AST TRUST BOUNDARY", 
        fontsize=9, fontweight='bold', color=BORDER_GREEN, fontfamily='sans-serif')

# Inner Box A: Rigid AST Schema & Column Validator
draw_box(1.1, 1.6, 5.0, 0.7, "Rigid AST Schema & Column Validator", bg_col=BOX_GREEN, border_col=BORDER_GREEN)
draw_arrow(3.6, 1.6, 3.6, 1.3, color=BORDER_GREEN)

# Inner Box B: Deterministic Pandas Execution Engine
draw_box(1.1, 0.75, 5.0, 0.7, "Deterministic Pandas Execution Engine", bg_col=BOX_GREEN, border_col=BORDER_GREEN)

# Arrow from Deterministic Pandas Engine to 4-Part Response (Exact Numbers)
# Starts at right of Pandas engine (x=6.1, y=1.1) -> Horizontal to x=11.1 -> Vertical up to y=1.9
ax.plot([6.1, 11.1], [1.1, 1.1], color=BORDER_GREEN, lw=1.6)
draw_arrow(11.1, 1.1, 11.1, 1.9, color=BORDER_GREEN)
ax.text(11.8, 1.35, "Exact Numbers", ha='center', va='center', fontsize=8.5, 
        color=BORDER_GREEN, fontweight='bold', backgroundcolor=EXCALI_BG)

# 6. 4-PART UI RESPONSE (Center/Bottom Right)
resp_bg = patches.FancyBboxPatch((8.5, 1.9), 5.2, 2.5, boxstyle="round,pad=0.1,rounding_size=0.18",
                                facecolor=BOX_AMBER, edgecolor=BORDER_AMBER, linewidth=1.8)
ax.add_patch(resp_bg)
ax.text(11.1, 4.05, "4-PART UI RESPONSE", ha='center', va='center', 
        fontsize=11, fontweight='bold', color=TEXT_COLOR, fontfamily='sans-serif')

bullet_items = [
    "-  Factual Direct Answer",
    "-  Formatted Evidence Table",
    "-  Interactive Plotly Viz",
    "-  Verifiable Audit Trail"
]
for i, item in enumerate(bullet_items):
    ax.text(9.0, 3.55 - (i * 0.42), item, ha='left', va='center', 
            fontsize=9.5, color=SUBTEXT_COLOR, fontfamily='sans-serif')

plt.tight_layout()
output_path = "architecture_trust_boundary.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=EXCALI_BG)
plt.close()
print(f"Successfully generated {output_path}")
