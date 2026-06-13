(function () {
  var pyodide = null;
  var ready = false;
  var appRoot = "/tmp/m2c-browser-root";

  var sourceEl = document.getElementById("source");
  var contextEl = document.getElementById("context");
  var outputEl = document.getElementById("output");
  var graphEl = document.getElementById("output-graph");
  var buttonEl = document.getElementById("decompile");
  var visualizeEl = document.getElementById("visualize");
  var functionEl = document.getElementById("function");
  var regvarsSelectEl = document.getElementById("regvars-select");
  var regvarsEl = document.getElementById("regvars");
  var formEl = document.getElementsByTagName("form")[0];
  var darkModeCheckbox = document.getElementById("dark");

  var optionIds = [
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
    if (err && err.stack) {
      return err.stack;
    }
    if (err && err.message) {
      return err.message;
    }
    if (err && typeof err === "object") {
      try {
        return JSON.stringify(err);
      } catch (jsonErr) {
        return String(err);
      }
    }
    return String(err);
  }

  function updateFunctions() {
    var previous = functionEl.value;
    functionEl.innerHTML = "";

    var allOption = document.createElement("option");
    allOption.value = "all";
    allOption.textContent = "all functions";
    functionEl.appendChild(allOption);

    var matches = sourceEl.value.matchAll(/^\s*(?:(?:glabel|dlabel|arm_func_start|thumb_func_start|non_word_aligned_thumb_func_start|ARM_FUNC_START|THUMB_FUNC_START|NON_WORD_ALIGNED_THUMB_FUNC_START)\s+(\S+)|\.fn\s+(\S+)|([A-Za-z_.$][\w.$]*):)/gm);
    for (var match of matches) {
      var name = match[1] || match[2] || match[3];
      var option = document.createElement("option");
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

  function saveState() {
    var options = {};

    for (var id of optionIds) {
      var el = document.getElementById(id);
      options[el.name || id] = el.type === "checkbox" ? (el.checked ? "yes" : "no") : el.value;
    }

    localStorage.mips_to_c_saved_source = sourceEl.value;
    localStorage.mips_to_c_saved_context = contextEl.value;
    localStorage.mips_to_c_saved_options = JSON.stringify(options);
  }

  function restoreState() {
    var savedSource = localStorage.mips_to_c_saved_source;
    var savedContext = localStorage.mips_to_c_saved_context;
    var savedOptions = localStorage.mips_to_c_saved_options;

    if (savedSource) sourceEl.value = savedSource;
    if (savedContext) contextEl.value = savedContext;

    if (!savedOptions) {
      return;
    }
    try {
      var options = JSON.parse(savedOptions);
      for (var key in options) {
        var el = document.getElementsByName(key)[0];
        if (!el) {
          continue;
        }
        if (el.type === "checkbox") {
          el.checked = options[key] === "yes";
        } else {
          el.value = options[key];
        }
      }
    } catch (err) {
      console.warn("Unable to restore saved m2c browser state", err);
    }
  }

  function updateDarkMode() {
    document.documentElement.className = darkModeCheckbox.checked ? "dark-theme" : "";
  }

  function showTextOutput(value) {
    graphEl.style.display = "none";
    outputEl.style.display = "";
    graphEl.replaceChildren();
    outputEl.value = value;
  }

  function showGraphOutput(svgElement) {
    outputEl.style.display = "none";
    outputEl.value = "";
    graphEl.style.display = "block";
    graphEl.replaceChildren(svgElement);
  }

  function normalizeDotForBrowser(dotSource) {
    return dotSource
      .replace(
        'node [shape="rect", fontname="Monospace"];',
        'node [shape="rect", fontname="Courier", margin="0.12,0.08"];'
      )
      .replace('edge [fontname="Monospace"];', 'edge [fontname="Courier"];');
  }

  function buildFlags() {
    var flags = [];
    var globals = document.getElementById("globals").value;
    var target = document.getElementById("target").value;
    var commentStyle = document.getElementById("comment-style").value;
    var regvarsSelect = regvarsSelectEl.value;

    flags.push("--globals", globals);
    flags.push("--target", target);

    if (commentStyle === "none") {
      flags.push("--comment-style=none");
    } else if (commentStyle.indexOf("oneline") === 0) {
      flags.push("--comment-style=oneline");
    } else {
      flags.push("--comment-style=multiline");
    }

    if (commentStyle.indexOf("unaligned") !== -1) {
      flags.push("--comment-column=0");
    }

    if (functionEl.value && functionEl.value !== "all") {
      flags.push("--function", functionEl.value);
    }

    if (regvarsSelect === "saved" || regvarsSelect === "all") {
      flags.push("--reg-vars", regvarsSelect);
    } else if (regvarsSelect === "custom" && regvarsEl.value.trim()) {
      flags.push("--reg-vars", regvarsEl.value.trim());
    }

    if (document.getElementById("void").checked) flags.push("--void");
    if (document.getElementById("debug").checked) flags.push("--debug");
    if (document.getElementById("noandor").checked) flags.push("--no-andor");
    if (document.getElementById("nocasts").checked) flags.push("--no-casts");
    if (document.getElementById("allman").checked) flags.push("--allman");
    if (document.getElementById("knr").checked) flags.push("--knr");
    if (document.getElementById("extraswitchindent").checked) flags.push("--indent-switch-contents");
    if (document.getElementById("leftptr").checked) flags.push("--pointer-style", "left");
    if (document.getElementById("zfillconstants").checked) flags.push("--zfill-constants");
    if (document.getElementById("noifs").checked) flags.push("--gotos-only");
    if (document.getElementById("noswitches").checked) flags.push("--no-switches");
    if (document.getElementById("nounkinference").checked) flags.push("--no-unk-inference");
    if (document.getElementById("stackstructs").checked) flags.push("--stack-structs");
    if (document.getElementById("nostackspill").checked) flags.push("--no-stack-spill");
    if (document.getElementById("descendingregs").checked) flags.push("--descending-regs");
    if (document.getElementById("backwardsbss").checked) flags.push("--backwards-bss");

    return flags;
  }

  function writeBrowserFiles(files) {
    function mkdirp(path) {
      var parts = path.split("/");
      var current = "";
      for (var i = 0; i < parts.length; i += 1) {
        if (!parts[i]) {
          continue;
        }
        current += "/" + parts[i];
        if (pyodide.FS.analyzePath(current).exists) {
          continue;
        }
        try {
          pyodide.FS.mkdir(current);
        } catch (err) {
          if (!pyodide.FS.analyzePath(current).exists) {
            throw err;
          }
        }
      }
    }

    mkdirp(appRoot + "/m2c");
    mkdirp(appRoot + "/m2c_pycparser");
    mkdirp(appRoot + "/m2c_pycparser/ply");

    for (var path in files) {
      var fullPath = appRoot + "/" + path;
      var dirPath = fullPath.split("/").slice(0, -1).join("/");
      mkdirp(dirPath);
      pyodide.FS.writeFile(fullPath, files[path], { encoding: "utf8" });
    }
  }

  async function initPyodide() {
    try {
      if (!window.M2C_PYTHON_FILES) {
        throw new Error("m2c.generated.js was not loaded");
      }

      setBusyButton("decompile", "Loading...");
      pyodide = await loadPyodide();
      setBusyButton("decompile", "Installing...");
      writeBrowserFiles(window.M2C_PYTHON_FILES);
      await pyodide.runPythonAsync("import sys\nsys.path.insert(0, '/tmp/m2c-browser-root')\nfrom m2c.browser import decompile_from_json\n");
      ready = true;
      buttonEl.disabled = false;
      visualizeEl.disabled = false;
      resetButtonLabels();
      var autorun = new URLSearchParams(window.location.search).get("autorun");
      if (autorun !== null) {
        runM2c(autorun === "visualize" ? "visualize" : "decompile");
      }
    } catch (err) {
      console.error(err);
      var message = formatError(err);
      setBusyButton("decompile", "Failed");
      showTextOutput(message);
    }
  }

  async function runM2c(action) {
    if (!ready) {
      return;
    }

    saveState();
    buttonEl.disabled = true;
    visualizeEl.disabled = true;
    showTextOutput("");
    setBusyButton(action, action === "visualize" ? "Visualizing..." : "Decompiling...");

    try {
      var flags = buildFlags();
      if (action === "visualize") {
        flags.push("--visualize");
      }

      pyodide.globals.set("m2c_options_json", JSON.stringify({
        source: sourceEl.value,
        context: contextEl.value,
        flags: flags
      }));
      var result = null;
      var returncode;
      var output;
      var outputType;
      try {
        result = await pyodide.runPythonAsync("decompile_from_json(m2c_options_json).copy()");
        returncode = result.get("returncode");
        output = result.get("output");
        outputType = result.get("output_type");
      } finally {
        if (result) {
          result.destroy();
        }
      }
      document.body.dataset.m2cReturncode = String(returncode);
      document.body.dataset.m2cOutput = output;

      if (returncode === 0 && outputType === "dot") {
        if (!window.m2cVizReady) {
          throw new Error("Viz.js was not loaded");
        }
        setBusyButton("visualize", "Rendering...");
        var viz = await window.m2cVizReady;
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

  restoreState();
  if (!localStorage.mips_to_c_saved_options || localStorage.mips_to_c_saved_options.indexOf("\"dark\"") === -1) {
    darkModeCheckbox.checked = window.matchMedia("prefers-color-scheme: dark").matches;
  }
  updateFunctions();
  updateRegvars();
  updateDarkMode();

  sourceEl.addEventListener("blur", function () {
    updateFunctions();
    saveState();
  });
  sourceEl.addEventListener("change", saveState);
  contextEl.addEventListener("change", saveState);
  document.getElementById("options").addEventListener("change", function () {
    updateRegvars();
    updateDarkMode();
    saveState();
  });
  formEl.addEventListener("submit", function (event) {
    event.preventDefault();
    runM2c(event.submitter && event.submitter.id === "visualize" ? "visualize" : "decompile");
  });

  initPyodide();
})();
