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
  var browserPython = String.raw`
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import List, TypedDict

from .main import parse_flags, run


class BrowserResult(TypedDict):
    returncode: int
    output: str


def decompile(source: str, context: str, flags: List[str]) -> BrowserResult:
    stdout = io.StringIO()
    stderr = io.StringIO()
    is_visualize = False

    try:
        with tempfile.TemporaryDirectory(prefix="m2c-browser-") as tmpdir:
            base_path = Path(tmpdir)
            asm_path = base_path / "input.s"
            asm_path.write_text(source, encoding="utf-8")

            argv = list(flags)
            argv.append("--no-cache")
            argv.append("--visualize-format=dot")

            if context:
                context_path = base_path / "context.c"
                context_path.write_text(context, encoding="utf-8")
                argv.extend(["--context", str(context_path)])

            argv.append(str(asm_path))

            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                options = parse_flags(argv)
                is_visualize = options.visualize_flowgraph is not None
                returncode = run(options)
    except SystemExit as exc:
        is_visualize = False
        returncode = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        is_visualize = False
        returncode = 1
        stdout.write(f"Internal browser wrapper error:\n{exc}\n")

    err = stderr.getvalue()
    if err and (not is_visualize or returncode != 0):
        stdout.write(err)
    output = stdout.getvalue()
    if is_visualize and returncode == 0:
        dot_start = output.find("digraph {")
        if dot_start != -1:
            output = output[dot_start:]

    return {
        "returncode": returncode,
        "output": output,
    }


def decompile_from_json(options_json: str) -> BrowserResult:
    options = json.loads(options_json)
    if not isinstance(options, dict):
        return {
            "returncode": 1,
            "output": "Expected browser options JSON to decode to an object.\n",
        }

    source = options.get("source")
    context = options.get("context")
    flags = options.get("flags")
    if not isinstance(source, str) or not isinstance(context, str) or not isinstance(
        flags, list
    ):
        return {
            "returncode": 1,
            "output": "Expected browser options to contain source, context, and flags.\n",
        }
    return decompile(source, context, [str(flag) for flag in flags])
`;
  var functionStartDirectives = [
    "glabel",
    "arm_func_start",
    "thumb_func_start",
    "non_word_aligned_thumb_func_start",
    "ARM_FUNC_START",
    "THUMB_FUNC_START",
    "NON_WORD_ALIGNED_THUMB_FUNC_START",
    ".fn"
  ];
  var localLabelRe = /^(?:loc_|locret_|def_|lbl_|LAB_|switchD_|jump_|_[0-9A-Fa-f]{7,8}(?:_.*)?$)/;

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
    var message;
    if (err && err.stack) {
      message = err.stack;
    } else if (err && err.message) {
      message = err.message;
    } else if (err && typeof err === "object") {
      try {
        message = JSON.stringify(err);
      } catch (jsonErr) {
        message = String(err);
      }
    } else {
      message = String(err);
    }
    return message.indexOf("Error") === -1 && message.indexOf("error") === -1 ? "Error: " + message : message;
  }

  function stripLineComment(line) {
    return line.replace(/[#;@].*$/, "").replace(/\/\/.*$/, "").trim();
  }

  function parseFunctionName(line) {
    var stripped = stripLineComment(line);
    var parts = stripped.split(/\s+/);
    var directive = parts[0];
    if (functionStartDirectives.indexOf(directive) !== -1 && parts[1]) {
      return parts[1].replace(/,$/, "");
    }

    var labelMatch = stripped.match(/^([a-zA-Z0-9_.$]+|"[a-zA-Z0-9_.$<>@,-]+"):/);
    if (!labelMatch) {
      return "";
    }
    var label = labelMatch[1].replace(/^"|"$/g, "");
    if (label.charAt(0) === "." || localLabelRe.test(label)) {
      return "";
    }
    return label;
  }

  function hasFunctionStart(source) {
    return source.split(/\r?\n/).some(function (line) {
      return parseFunctionName(line) !== "";
    });
  }

  function sourceWithDefaultFunction() {
    return hasFunctionStart(sourceEl.value) ? sourceEl.value : "glabel foo\n" + sourceEl.value;
  }

  function updateFunctions() {
    var previous = functionEl.value;
    functionEl.innerHTML = "";

    var allOption = document.createElement("option");
    allOption.value = "all";
    allOption.textContent = "all functions";
    functionEl.appendChild(allOption);

    for (var line of sourceEl.value.split(/\r?\n/)) {
      var name = parseFunctionName(line);
      if (!name) {
        continue;
      }
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

    localStorage.m2c_saved_source = sourceEl.value;
    localStorage.m2c_saved_context = contextEl.value;
    localStorage.m2c_saved_options = JSON.stringify(options);
  }

  function restoreState() {
    var savedSource = localStorage.m2c_saved_source;
    var savedContext = localStorage.m2c_saved_context;
    var savedOptions = localStorage.m2c_saved_options;

    if (savedSource) sourceEl.value = savedSource;
    if (savedContext) contextEl.value = savedContext;

    if (!savedOptions) {
      return {};
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
      return options;
    } catch (err) {
      console.warn("Unable to restore saved m2c browser state", err);
      return {};
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
    return dotSource
      .replace(
        /fontname="Monospace"/g,
        'fontname="Courier"'
      )
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

    pyodide.FS.writeFile(appRoot + "/m2c/browser.py", browserPython, { encoding: "utf8" });
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement("script");
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
        throw new Error("m2c.generated.js was not loaded");
      }

      setBusyButton("decompile", "Loading...");
      await loadScript(window.M2C_VENDOR_PATHS.pyodideScript);
      pyodide = await loadPyodide({
        indexURL: window.M2C_VENDOR_PATHS.pyodideIndexURL
      });
      setBusyButton("decompile", "Installing...");
      writeBrowserFiles(window.M2C_PYTHON_FILES);
      await pyodide.runPythonAsync("import json\nimport sys\nsys.path.insert(0, '/tmp/m2c-browser-root')\nsys.setrecursionlimit(min(2**31 - 1, 10 * sys.getrecursionlimit()))\nfrom m2c.browser import decompile_from_json\n");
      ready = true;
      buttonEl.disabled = false;
      visualizeEl.disabled = false;
      resetButtonLabels();
      var autorun = new URLSearchParams(window.location.search).get("autorun");
      if (autorun !== null) {
        runM2c(autorun === "visualize");
      }
    } catch (err) {
      console.error(err);
      var message = formatError(err);
      setBusyButton("decompile", "Failed");
      showTextOutput(message);
    }
  }

  async function runM2c(visualize) {
    if (!ready) {
      return;
    }

    saveState();
    buttonEl.disabled = true;
    visualizeEl.disabled = true;
    clearOutput();
    setBusyButton(visualize ? "visualize" : "decompile", visualize ? "Visualizing..." : "Decompiling...");

    try {
      var flags = buildFlags();
      if (visualize) {
        flags.push("--visualize");
      }

      pyodide.globals.set("m2c_options_json", JSON.stringify({
        source: sourceWithDefaultFunction(),
        context: contextEl.value,
        flags: flags
      }));
      var result = JSON.parse(await pyodide.runPythonAsync("json.dumps(decompile_from_json(m2c_options_json))"));
      var returncode = result.returncode;
      var output = result.output;

      if (returncode === 0 && visualize) {
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

  var restoredOptions = restoreState();
  clearOutput();
  if (!("dark" in restoredOptions)) {
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
    runM2c(event.submitter && event.submitter.id === "visualize");
  });

  initPyodide();
})();
