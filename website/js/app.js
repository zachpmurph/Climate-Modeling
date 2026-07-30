/* Flood Explorer boot: tab routing + module init. */

import { initAtlas } from "./atlas.js";
import { initPlayground } from "./playground.js";
import { initValidation } from "./validation.js";

function initTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", t === tab ? "true" : "false");
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `tab-${tab.dataset.tab}`);
      });
    });
  });
}

initTabs();
initAtlas();
initPlayground();
initValidation();
