(function () {
  let pyodide = null;
  let ready = false;
  const appRoot = "/tmp/m2c-browser-root";

  const sourceEl = document.getElementById("source");
  const contextEl = document.getElementById("context");
  const outputEl = document.getElementById("output");
  const graphEl = document.getElementById("output-graph");
  const buttonEl = document.getElementById("decompile");
  const visualizeEl = document.getElementById("visualize");
  const functionEl = document.getElementById("function");
  const regvarsSelectEl = document.getElementById("regvars-select");
  const regvarsEl = document.getElementById("regvars");
  const formEl = document.getElementsByTagName("form")[0];
  const darkModeCheckbox = document.getElementById("dark");
  const browserPython = String.raw`
from __future__ import annotations

import json

from m2c.main import BrowserResult, decompile_for_browser


def decompile_from_json(options_json: str) -> str:
    try:
        options = json.loads(options_json)
        assert isinstance(options, dict)
        source = options["source"]
        context = options["context"]
        flags = options["flags"]
        assert isinstance(source, str)
        assert isinstance(context, str)
        assert isinstance(flags, list)
        assert all(isinstance(flag, str) for flag in flags)
        result = decompile_for_browser(source, context, flags)
    except (Exception, SystemExit) as exc:
        result = BrowserResult(1, f"Internal browser wrapper error:\n{exc}\n")
    return json.dumps({"returncode": result.returncode, "output": result.output})
`;
  // Keep in sync with the function-start directives in m2c/asm_file.py (the
  // .fn directive and the tuple at the end of parse_file).
  const functionStartDirectives = [
    "glabel",
    "arm_func_start",
    "thumb_func_start",
    "non_word_aligned_thumb_func_start",
    "ARM_FUNC_START",
    "THUMB_FUNC_START",
    "NON_WORD_ALIGNED_THUMB_FUNC_START",
    ".fn"
  ];
  // Keep these in sync with re_local_label and re_label in m2c/asm_file.py.
  const reLocalLabel = /^(?:loc_|locret_|def_|lbl_|LAB_|switchD_|jump_|LF?[0-9]+$|_[0-9A-Fa-f]{7,8}(?:_.*)?$)/;
  const reLabel = /^(?:([a-zA-Z0-9_.$]+)|"([a-zA-Z0-9_.$<>@,-]+)"):/;
  const reFunctionDirective = /^(\S+)\s+([^,\s]+)/;

  const optionIds = [
    "globals",
    "target",
    "comment-style",
    "regvars-select",
    "regvars",
    "void",
    "debug",
    "noandor",
    "nocasts",
    "allman",
    "knr",
    "extraswitchindent",
    "leftptr",
    "zfillconstants",
    "noifs",
    "noswitches",
    "nounkinference",
    "stackstructs",
    "nostackspill",
    "descendingregs",
    "backwardsbss",
    "dark"
  ];

  function resetButtonLabels() {
    buttonEl.value = "Decompile";
    visualizeEl.value = "Visualize";
  }

  function setBusyButton(action, message) {
    resetButtonLabels();
    if (action === "visualize") {
      visualizeEl.value = message;
    } else {
      buttonEl.value = message;
    }
  }

  function formatError(err) {
    let message = "Error: ";
    if (err && err.message) {
      message += err.message;
    } else if (err && typeof err === "object") {
      try {
        message += JSON.stringify(err);
      } catch (jsonErr) {
        message += String(err);
      }
    } else {
      message += String(err);
    }
    if (err && err.stack) {
      message += "\n\n" + err.stack;
    }
    return message;
  }

  // Mirror m2c's comment handling (see re_comment_or_string in m2c/asm_file.py):
  // strip comments but keep quoted strings intact, since those may contain
  // comment syntax. Note that `!` is a comment char for sh2.
  const reCommentOrString = /("(?:\\.|[^\\"])*"|[#;@!].*|\/\*.*?\*\/)/g;

  function getFunctionNames(source) {
    const names = [];
    for (const line of source.split("\n")) {
      let stripped = line
        .replace(reCommentOrString, function (match, quoted) {
          return quoted.charAt(0) === '"' ? quoted : " ";
        })
        .trim();
      let labelMatch;
      while ((labelMatch = stripped.match(reLabel)) !== null) {
        const label = labelMatch[1] || labelMatch[2];
        stripped = stripped.slice(labelMatch[0].length).trim();
        if (label.charAt(0) !== "." && !reLocalLabel.test(label)) {
          names.push(label);
        }
      }
      const directiveMatch = stripped.match(reFunctionDirective);
      if (directiveMatch && functionStartDirectives.includes(directiveMatch[1])) {
        names.push(directiveMatch[2]);
      }
    }
    return names;
  }

  function updateFunctions() {
    const previous = functionEl.value;
    functionEl.replaceChildren();

    const allOption = document.createElement("option");
    allOption.value = "all";
    allOption.textContent = "all functions";
    functionEl.appendChild(allOption);

    for (const name of getFunctionNames(sourceEl.value)) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      functionEl.appendChild(option);
    }

    functionEl.value = previous || "all";
    if (!functionEl.value) {
      functionEl.value = "all";
    }
  }

  function updateRegvars() {
    document.body.setAttribute("data-regvars", regvarsSelectEl.value);
    if (regvarsSelectEl.value === "custom" && !regvarsEl.value) {
      regvarsEl.value = "s0,s1,s2";
    }
  }

  function saveSource() {
    localStorage.m2c_saved_source = sourceEl.value;
  }

  function saveContext() {
    localStorage.m2c_saved_context = contextEl.value;
  }

  function saveOptions() {
    const options = {};

    for (const id of optionIds) {
      const el = document.getElementById(id);
      options[id] = el.type === "checkbox" ? (el.checked ? "yes" : "no") : el.value;
    }

    localStorage.m2c_saved_options = JSON.stringify(options);
  }

  function restoreState() {
    const savedSource = localStorage.m2c_saved_source;
    const savedContext = localStorage.m2c_saved_context;
    const savedOptions = localStorage.m2c_saved_options;

    if (savedSource) sourceEl.value = savedSource;
    if (savedContext) contextEl.value = savedContext;

    if (!savedOptions) {
      return {};
    }
    try {
      const options = JSON.parse(savedOptions);
      for (const key in options) {
        const el = document.getElementById(key);
        if (!el) {
          continue;
        }
        if (el.type === "checkbox") {
          el.checked = options[key] === "yes";
        } else {
          el.value = options[key];
        }
      }
      return options;
    } catch (err) {
      console.warn("Unable to restore saved m2c browser state", err);
    }
    return {};
  }

  function updateDarkMode() {
    document.documentElement.className = darkModeCheckbox.checked ? "dark-theme" : "";
  }

  function showTextOutput(value) {
    graphEl.style.display = "none";
    outputEl.style.display = "";
    graphEl.replaceChildren();
    outputEl.value = value;
    outputEl.focus();
  }

  function clearOutput() {
    graphEl.style.display = "none";
    outputEl.style.display = "";
    graphEl.replaceChildren();
    outputEl.value = "";
  }

  function showGraphOutput(svgElement) {
    outputEl.style.display = "none";
    outputEl.value = "";
    graphEl.style.display = "block";
    graphEl.replaceChildren(svgElement);
  }

  function normalizeDotForBrowser(dotSource) {
    // Force font to ensure boxes are drawn large enough for contents
    return dotSource.replace(
      /fontname="Monospace"/g,
      'fontname="Courier"'
    );
  }

  function buildFlags() {
    const flags = [];
    const globals = document.getElementById("globals").value;
    const target = document.getElementById("target").value;
    const commentStyle = document.getElementById("comment-style").value;
    const regvarsSelect = regvarsSelectEl.value;

    flags.push("--globals=" + globals);
    flags.push("--target=" + target);

    if (commentStyle === "none") {
      flags.push("--comment-style=none");
    } else if (commentStyle.startsWith("oneline")) {
      flags.push("--comment-style=oneline");
    } else {
      flags.push("--comment-style=multiline");
    }

    if (commentStyle.includes("unaligned")) {
      flags.push("--comment-column=0");
    }

    if (functionEl.value && functionEl.value !== "all") {
      flags.push("--function=" + functionEl.value);
    }

    if (regvarsSelect === "saved" || regvarsSelect === "all") {
      flags.push("--reg-vars=" + regvarsSelect);
    } else if (regvarsSelect === "custom" && regvarsEl.value.trim()) {
      flags.push("--reg-vars=" + regvarsEl.value.trim());
    }

    const boolFlags = {
      void: "--void", debug: "--debug", noandor: "--no-andor",
      nocasts: "--no-casts", allman: "--allman", knr: "--knr",
      extraswitchindent: "--indent-switch-contents",
      leftptr: "--pointer-style=left", zfillconstants: "--zfill-constants",
      noifs: "--gotos-only", noswitches: "--no-switches",
      nounkinference: "--no-unk-inference", stackstructs: "--stack-structs",
      nostackspill: "--no-stack-spill", descendingregs: "--descending-regs",
      backwardsbss: "--backwards-bss"
    };
    for (const id in boolFlags) {
      if (document.getElementById(id).checked) flags.push(boolFlags[id]);
    }

    return flags;
  }

  function writeBrowserFiles(files) {
    function mkdirp(path) {
      const parts = path.split("/");
      let current = "";
      for (const part of parts) {
        if (!part) {
          continue;
        }
        current += "/" + part;
        if (!pyodide.FS.analyzePath(current).exists) {
          pyodide.FS.mkdir(current);
        }
      }
    }

    for (const path in files) {
      const fullPath = appRoot + "/" + path;
      const dirPath = fullPath.split("/").slice(0, -1).join("/");
      mkdirp(dirPath);
      pyodide.FS.writeFile(fullPath, files[path], { encoding: "utf8" });
    }
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      const script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  async function initPyodide() {
    try {
      if (!window.M2C_VENDOR_PATHS) {
        throw new Error("vendor-paths.js was not loaded");
      }
      if (!window.M2C_PYTHON_FILES) {
        throw new Error("m2c.js was not loaded");
      }

      setBusyButton("decompile", "Loading...");
      await loadScript(window.M2C_VENDOR_PATHS.pyodideScript);
      pyodide = await loadPyodide({
        indexURL: window.M2C_VENDOR_PATHS.pyodideIndexURL
      });
      setBusyButton("decompile", "Installing...");
      writeBrowserFiles(window.M2C_PYTHON_FILES);
      await pyodide.runPythonAsync(
        "import sys\nsys.path.insert(0, '/tmp/m2c-browser-root')\n"
        + "sys.setrecursionlimit(min(2**31 - 1, 10 * sys.getrecursionlimit()))\n"
      );
      await pyodide.runPythonAsync(browserPython);
      ready = true;
      buttonEl.disabled = false;
      visualizeEl.disabled = false;
      resetButtonLabels();
      const autorun = new URLSearchParams(window.location.search).get("autorun");
      if (autorun !== null) {
        runM2c(autorun === "visualize");
      }
    } catch (err) {
      console.error(err);
      const message = formatError(err);
      setBusyButton("decompile", "Failed");
      showTextOutput(message);
    }
  }

  async function runM2c(visualize) {
    if (!ready) {
      return;
    }

    buttonEl.disabled = true;
    visualizeEl.disabled = true;
    clearOutput();
    setBusyButton(visualize ? "visualize" : "decompile", visualize ? "Visualizing..." : "Decompiling...");

    try {
      const flags = buildFlags();
      if (visualize) {
        flags.push("--visualize");
      }

      const source = getFunctionNames(sourceEl.value).length
        ? sourceEl.value
        : "glabel foo\n" + sourceEl.value;
      pyodide.globals.set("m2c_options_json", JSON.stringify({
        source: source,
        context: contextEl.value,
        flags: flags
      }));
      const result = JSON.parse(await pyodide.runPythonAsync("decompile_from_json(m2c_options_json)"));
      const returncode = result.returncode;
      const output = result.output;

      if (returncode === 0 && visualize) {
        setBusyButton("visualize", "Rendering...");
        const viz = await window.m2cVizReady;
        showGraphOutput(viz.renderSVGElement(normalizeDotForBrowser(output)));
      } else {
        showTextOutput(output);
      }
    } catch (err) {
      console.error(err);
      showTextOutput(formatError(err));
    } finally {
      resetButtonLabels();
      buttonEl.disabled = false;
      visualizeEl.disabled = false;
    }
  }

  const restoredOptions = restoreState();
  clearOutput();
  if (!("dark" in restoredOptions)) {
    darkModeCheckbox.checked = window.matchMedia("prefers-color-scheme: dark").matches;
  }
  updateFunctions();
  updateRegvars();
  updateDarkMode();

  sourceEl.addEventListener("blur", function () {
    updateFunctions();
    saveSource();
  });
  sourceEl.addEventListener("change", saveSource);
  contextEl.addEventListener("change", saveContext);
  document.getElementById("options").addEventListener("change", function () {
    updateRegvars();
    updateDarkMode();
    saveOptions();
  });
  formEl.addEventListener("submit", function (event) {
    event.preventDefault();
    runM2c(event.submitter && event.submitter.id === "visualize");
  });

  initPyodide();
})();