/**
 * Conversational Data Analysis Assistant - Frontend Interactions
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Collapsible Sidebar Logic
  const sidebar = document.getElementById("sidebar");
  const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
  const floatingToggleBtn = document.getElementById("floatingToggleBtn");

  function setSidebarCollapsed(collapsed) {
    if (collapsed) {
      sidebar.classList.add("collapsed");
      localStorage.setItem("sidebar_collapsed", "true");
    } else {
      sidebar.classList.remove("collapsed");
      localStorage.setItem("sidebar_collapsed", "false");
    }
  }

  // Restore sidebar state from localStorage
  const savedState = localStorage.getItem("sidebar_collapsed");
  if (savedState === "true") {
    setSidebarCollapsed(true);
  }

  if (sidebarToggleBtn) {
    sidebarToggleBtn.addEventListener("click", () => {
      setSidebarCollapsed(true);
    });
  }

  if (floatingToggleBtn) {
    floatingToggleBtn.addEventListener("click", () => {
      setSidebarCollapsed(false);
    });
  }

  // 2. Suggestion Chip Ingestion
  const suggestionChips = document.querySelectorAll(".suggestion-chip");
  const queryInput = document.getElementById("queryInput");

  suggestionChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const questionText = chip.getAttribute("data-query");
      if (queryInput && questionText) {
        queryInput.value = questionText;
        queryInput.focus();
      }
    });
  });

  // 3. Accordions
  const accordionHeaders = document.querySelectorAll(".accordion-header");
  accordionHeaders.forEach((header) => {
    header.addEventListener("click", () => {
      const content = header.nextElementSibling;
      const arrow = header.querySelector(".accordion-arrow");
      if (content) {
        const isHidden = content.style.display === "none" || !content.style.display;
        content.style.display = isHidden ? "block" : "none";
        if (arrow) {
          arrow.style.transform = isHidden ? "rotate(180deg)" : "rotate(0deg)";
        }
      }
    });
  });

  // 4. Analysis Form Submission Spinner
  const queryForm = document.getElementById("queryForm");
  const spinnerContainer = document.getElementById("spinnerContainer");
  const submitBtn = document.getElementById("submitQueryBtn");

  if (queryForm && spinnerContainer) {
    queryForm.addEventListener("submit", () => {
      spinnerContainer.style.display = "flex";
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerText = "Analyzing...";
      }
    });
  }
});
