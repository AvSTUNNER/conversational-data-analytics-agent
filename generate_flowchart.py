import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
fig.patch.set_facecolor('#0B0F19')
ax.set_facecolor('#0B0F19')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Title & Subtitle
ax.text(8, 8.4, "Dual-Engine Conversational Data Analytics Assistant", 
        ha='center', va='center', fontsize=20, fontweight='bold', color='#FFFFFF', fontfamily='sans-serif')
ax.text(8, 7.9, "Zero-Hallucination Architecture: Semantic RAG + Deterministic Pandas Execution with Strict AST Trust Boundary", 
        ha='center', va='center', fontsize=11, color='#94A3B8', fontfamily='sans-serif')

def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, badge=None):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.18",
                                 facecolor=bg_color, edgecolor=border_color, linewidth=1.8)
    ax.add_patch(rect)
    if badge:
        b_rect = patches.FancyBboxPatch((x + 0.15, y + h - 0.32), len(badge)*0.11 + 0.2, 0.22, 
                                        boxstyle="round,pad=0.03,rounding_size=0.08",
                                        facecolor=border_color, edgecolor='none')
        ax.add_patch(b_rect)
        ax.text(x + 0.15 + (len(badge)*0.11 + 0.2)/2, y + h - 0.21, badge,
                ha='center', va='center', fontsize=7.5, fontweight='bold', color='#FFFFFF')
        
    text_y = y + h/2 + 0.08 if subtitle else y + h/2
    ax.text(x + w/2, text_y, title, ha='center', va='center', 
            fontsize=10.5, fontweight='bold', color='#FFFFFF', fontfamily='sans-serif')
    if subtitle:
        ax.text(x + w/2, y + h/2 - 0.22, subtitle, ha='center', va='center', 
                fontsize=8.5, color='#CBD5E1', fontfamily='sans-serif')

def draw_arrow(x1, y1, x2, y2, label=None, color='#64748B', style='->', lw=1.8, label_offset=(0, 0.15)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=4, shrinkB=4))
    if label:
        mid_x = (x1 + x2) / 2 + label_offset[0]
        mid_y = (y1 + y2) / 2 + label_offset[1]
        ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8, 
                color=color, fontweight='bold', backgroundcolor='#0B0F19')

# 1. User Query & Ingestion
draw_box(0.6, 5.0, 2.5, 1.4, "User Query", "NL Prompt + Raw Dataset", "#1E293B", "#38BDF8", "INGESTION")

# 2. Query Router
draw_box(3.8, 5.0, 2.7, 1.4, "Semantic Query Router", "Intent Classifier (NLP)", "#1E293B", "#818CF8", "TRIAGE")
draw_arrow(3.1, 5.7, 3.8, 5.7, color="#38BDF8")

# Branch A: Pure Knowledge (Upward)
draw_arrow(5.15, 6.4, 5.15, 7.1, color="#A855F7")
draw_arrow(5.15, 7.1, 7.5, 7.1, "KNOWLEDGE", color="#A855F7", label_offset=(0, 0.18))
draw_box(7.5, 6.4, 3.2, 1.4, "Local RAG Vector Engine", "TF-IDF + Cosine Similarity", "#2E1065", "#C084FC", "SEMANTIC RAG")

# Branch B: Analytical Path (Center)
draw_arrow(6.5, 5.7, 7.5, 5.7, "ANALYTICAL", color="#38BDF8", label_offset=(0, 0.18))
draw_box(7.5, 4.6, 3.2, 1.4, "LLM Analysis Planner", "AnalysisPlan Schema (JSON)", "#0F2847", "#38BDF8", "PLANNING")

# Branch C: Knowledge Artifact Data Profiler (Bottom Left)
draw_box(3.8, 2.6, 2.7, 1.4, "Dataset Profiler", "Schema, Types & Stats", "#1E293B", "#0EA5E9", "OBSERVATION")
draw_arrow(1.85, 5.0, 1.85, 3.3, color="#64748B")
draw_arrow(1.85, 3.3, 3.8, 3.3, "Active DF", color="#64748B")
draw_arrow(5.15, 4.0, 5.15, 5.0, "Context", color="#0EA5E9")
draw_arrow(6.5, 3.3, 7.5, 3.3, color="#10B981")

# Trust Boundary Box (Enclosing Validator & Deterministic Execution)
tb_rect = patches.FancyBboxPatch((7.2, 0.6), 7.8, 3.6, boxstyle="round,pad=0.15,rounding_size=0.2",
                                facecolor='#061D1E', edgecolor='#10B981', linewidth=1.5, linestyle='--')
ax.add_patch(tb_rect)
ax.text(7.4, 4.0, "[TRUST BOUNDARY] NUMERICAL INTEGRITY & AST VALIDATION", 
        fontsize=8.5, fontweight='bold', color='#10B981', fontfamily='sans-serif')

# Strict Plan Validator
draw_box(7.5, 2.4, 3.2, 1.3, "Strict Plan Validator", "Column, Type & Op Verification", "#064E3B", "#34D399", "VALIDATOR")
draw_arrow(9.1, 4.6, 9.1, 3.7, "AnalysisPlan", color="#38BDF8")

# Deterministic Pandas Engine
draw_box(11.5, 2.4, 3.2, 1.3, "Pandas Calculation Engine", "Deterministic Aggregations", "#064E3B", "#10B981", "AUTHORITY")
draw_arrow(10.7, 3.05, 11.5, 3.05, "Approved IR", color="#34D399")

# RAG Knowledge grounded answer to output
draw_arrow(10.7, 7.1, 13.1, 7.1, color="#C084FC")
draw_arrow(13.1, 7.1, 13.1, 6.4, "Grounded Text", color="#C084FC")

# Analytical Result to Output
draw_arrow(13.1, 3.7, 13.1, 5.0, "Exact Values", color="#10B981")

# Output: 4-Part Response
draw_box(11.5, 5.0, 3.2, 1.4, "4-Part UI Response", "Answer + Table + Chart + Audit", "#1E293B", "#F59E0B", "DELIVERY")

plt.tight_layout()
plt.savefig("architecture_flowchart.png", dpi=300, bbox_inches='tight', facecolor='#0B0F19')
print("Successfully generated clean architecture_flowchart.png")
