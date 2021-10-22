#!/usr/bin/env python3
import tempfile
import cgi
import cgitb
import os
import string
import subprocess
import sys

# cgi tracebacks
cgitb.enable()


def print_headers(content_type: str) -> None:
    print(f"Content-Type: {content_type}; charset=utf-8")
    print()
    sys.stdout.flush()


form = cgi.FieldStorage()
if "source" in form:
    source = form["source"].value if "source" in form else ""
    context = form["context"].value if "context" in form else None
    if "glabel" not in source:
        source = "glabel foo\n" + source
    source = bytes(source, "utf-8")
    script_path = os.path.join(os.path.dirname(__file__), "mips_to_c.py")
    cmd = ["python3", script_path, "/dev/stdin"]
    if "debug" in form:
        cmd.append("--debug")
    if "void" in form:
        cmd.append("--void")
    if "noifs" in form:
        cmd.append("--no-ifs")
    if "noandor" in form:
        cmd.append("--no-andor")
    if "nocasts" in form:
        cmd.append("--no-casts")
    if "allman" in form:
        cmd.append("--allman")
    if "leftptr" in form:
        cmd.extend(["--pointer-style", "left"])
    if "globals" in form:
        value = form.getfirst("globals")
        if value in ("all", "used", "none"):
            cmd.extend(["--globals", value])
    if "visualize" in form:
        cmd.append("--visualize")
    if "compiler" in form:
        value = form.getfirst("compiler")
        if value in ("ido", "gcc"):
            cmd.extend(["--compiler", value])

    comment_style = form.getfirst("comment_style", "multiline")
    if "oneline" in comment_style:
        cmd.append("--comment-style=oneline")
    else:
        cmd.append("--comment-style=multiline")
    if "unaligned" in comment_style:
        cmd.append("--comment-column=0")

    function = form["functionselect"].value if "functionselect" in form else "all"
    FUNCTION_ALPHABET = string.ascii_letters + string.digits + "_"
    function = "".join(c for c in function if c in FUNCTION_ALPHABET)
    if function and function != "all":
        cmd.extend(["--function", function])

    regvars = ""
    if "regvarsselect" in form:
        sel = form["regvarsselect"].value
        if sel in ("saved", "most", "all"):
            regvars = sel
        elif sel == "custom":
            regvars = form.getvalue("regvars", "")
            REG_ALPHABET = string.ascii_lowercase + string.digits + ","
            regvars = "".join(c for c in regvars if c in REG_ALPHABET)

    if regvars:
        cmd.append("--reg-vars")
        cmd.append(regvars)
    try:
        if context:
            with tempfile.NamedTemporaryFile() as f:
                f.write(bytes(context, "utf-8"))
                f.file.close()
                # There's no need to do caching on the temporary context file
                cmd.extend(["--no-cache", "--context", f.name])
                res = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    input=source,
                    timeout=15,
                )
        else:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                input=source,
                timeout=15,
            )
    except:
        # Set the headers for the cgitb traceback
        print_headers(content_type="text/html")
        raise
    if "visualize" in form and res.returncode == 0:
        print_headers(content_type="image/svg+xml")
        print(res.stdout.decode("utf-8", "replace"))
    else:
        print_headers(content_type="text/html")
        print("<!DOCTYPE html><html>")
        if "dark" in form:
            print(
                """
<head>
<style>
body {
    background-color: #454545;
    color: #c0c0c0;
}
</style>
</head>
"""
            )
        print("<body><pre><plaintext>", end="") # assuming this is the result page, any chance of adding rough syntax highlighting?
        print(res.stdout.decode("utf-8", "replace"))
elif "?go" in os.environ.get("REQUEST_URI", ""):
    print_headers(content_type="text/html")
else:
    print_headers(content_type="text/html")
    print(
        """
<!DOCTYPE html><html>
<head>

	<style>
	* {
			box-sizing: border-box;
	}
	div{
			padding:16px
	}
	html, body, form, .main, .sidebar {
			margin: 0;
			height: 100%;
	}
	textarea, .sidebar iframe {
			width: 100%;/*calc(100% - 16px);*/
			height: calc(100% - (1em + 16px));
			border: 1px solid #888;
			m//argin: 8px ;
	}

	label {
			white-space: nowrap;
	}
	[data-regvars]:not([data-regvars=custom]) [name=regvars] {
			display: none;
	}
	[name=regvars] {
			width: 150px;
	}
	.main, .sidebar {
			display: flex;
			flex-direction: column;
			padding: 10px;
			margin:0px 4px;
	}
	.sidebar {
			display: none;
	}
	.has-sidebar form {
			display: flex;
			flex-direction: row;
	}
	.has-sidebar .main, .has-sidebar .sidebar {
			flex: 1;
	}
	.has-sidebar .sidebar {
			display: flex;
	}
	.dark-theme {
			background-color: #000;
			color: #ccc;
	}
	.dark-theme div,
	.dark-theme body,
	.dark-theme form,
	.dark-theme textarea {
			background-color: rgba(255, 255, 255, 0.06225);
			color: #c0c0c0;
	}
	.dark-theme div
	{
			box-shadow: 1px 1px 10px 0px black;
	}
	#options
	{
			overflow:scroll;box-shadow:none;background-color:inherit;display:flex;flex-wrap: wrap; flex:1; flex-basis: 300px;
	}
	input[type=submit]
	{
			color:white;
			Background-color:#282;
			border:2px #282 outset;
			font-weight:bold;
			float:right;
	}
	input[type=submit]:focus {
		border:2px #4f4 outset;
	}
	select
	{

	}
	label
	{
		flex:1;flex-basis: 250px;
	}
	.keyword{color:blue;}
	.opperator{color:grey}
	</style>
</head>
<body>
<form action="?go" method="post">
<div class="main">
  <div style="flex: 15;">
    MIPS assembly:
		<br>
    <textarea name="source" spellcheck="false"></textarea>
  </div>
  <br>
  <div style="flex: 11;">  
    Existing C source, preprocessed (optional):
    <br>
    <textarea name="context" spellcheck="false"></textarea>
  </div>
  <br>
  <div>
    <input type="submit" value="Decompile">
    <input type="submit" name="visualize" value="Visualize">
    <details open="">
    <summary>Options</summary>
			<div id="options">
				<label>Function: 
					<select name="functionselect">
					</select>
				</label>
				
				<label>Global declarations:
					<select name="globals">
						<option value="used">used</option>
						<option value="all">all</option>
						<option value="none">none</option>
					</select>
				</label>
				<label>Original Compiler:
					<select name="compiler">
						<option value="ido">ido</option>
						<option value="gcc">gcc</option>
					</select>
				</label>
				<label>Comment style:
					<select name="comment_style">
						<option value="multiline">/* ... */, aligned</option>
						<option value="multiline_unaligned">/* ... */, unaligned</option>
						<option value="oneline">// ..., aligned</option>
						<option value="oneline_unaligned">// ..., unaligned</option>
					</select>
				</label>
				<label>Use single var for:
					<select name="regvarsselect">
						<option value="none">none</option>
						<option value="saved">saved regs</option>
						<option value="all">all regs</option>
						<option value="custom">custom</option>
					</select>
				</label>
				<input name="regvars" value="">
				<br>				
				<label><input type="checkbox" name="void">Force void return type</label>
				<label><input type="checkbox" name="debug">Debug info</label>
				<label><input type="checkbox" name="noandor">Disable &amp;&amp;/||</label>
				<label><input type="checkbox" name="nocasts">Hide type casts</label>
				<label><input type="checkbox" name="allman">Allman braces</label>
				<label><input type="checkbox" name="leftptr">* to the left</label>
				<label><input type="checkbox" name="noifs">Use gotos for everything</label>
				<label><input type="checkbox" name="usesidebar">Output sidebar</label>
				<label><input type="checkbox" name="dark">Dark mode</label>
				<p>Notes:<br> to use a goto for a single branch, add "# GOTO" to the asm</p>				
			</div>
		</details>
  </div>
</div>
<div class="sidebar">
    Output:
    <br>
    <iframe src="about:blank" name="outputframe"></iframe>
</div>
<script>
var formEl = document.getElementsByTagName("form")[0]
var sourceEl = document.getElementsByName("source")[0];
var contextEl = document.getElementsByName("context")[0];
var sidebarCheckboxEl = document.getElementsByName("usesidebar")[0];
var savedSource = localStorage.mips_to_c_saved_source;
var savedContext = localStorage.mips_to_c_saved_context;
var savedOptions = localStorage.mips_to_c_saved_options;
if (savedSource) sourceEl.value = savedSource;
if (savedContext) contextEl.value = savedContext;
if (savedOptions) {
    savedOptions = JSON.parse(savedOptions);
    for (var key in savedOptions) {
        var el = document.getElementsByName(key)[0], val = savedOptions[key];
        if (el) {
            if (el.type === "checkbox")
                el.checked = val === "yes";
            else
                el.value = val;
        }
    }
}

var darkModeCheckbox = document.getElementsByName("dark")[0];
if (!savedOptions || !("dark" in savedOptions)) {
    darkModeCheckbox.checked = window.matchMedia("prefers-color-scheme: dark").matches;
}

function updateDarkMode() {
    document.documentElement.className = darkModeCheckbox.checked ? "dark-theme" : "";
}
updateDarkMode();
darkModeCheckbox.addEventListener("change", updateDarkMode);

var functionSelect = document.getElementsByName("functionselect")[0];
function updateFunctions() {
    var prevValue = functionSelect.value;
    functionSelect.innerHTML = "<option value='all'>all functions</option>";
    for (let match of sourceEl.value.matchAll(/^\\s*glabel\\s+([A-Za-z0-9_]+)/mg)) {
        var name = match[1];

        var option = document.createElement("option");
        option.value = name;
        option.innerText = name;
        option.selected = (name == prevValue);
        functionSelect.appendChild(option);
    }
}
updateFunctions();
functionSelect.style.width = getComputedStyle(functionSelect).width;
sourceEl.addEventListener("blur", updateFunctions);

var regVarsSelect = document.getElementsByName("regvarsselect")[0];
function updateRegVars(e) {
    document.body.setAttribute("data-regvars", regVarsSelect.value);
    if (regVarsSelect.value === "custom" && e) {
        document.getElementsByName("regvars")[0].value = "s0,s1,s2";
    }
}
updateRegVars();
regVarsSelect.addEventListener("change", updateRegVars);

sourceEl.addEventListener("change", function() {
    localStorage.mips_to_c_saved_source = sourceEl.value;
});
contextEl.addEventListener("change", function() {
    localStorage.mips_to_c_saved_context = contextEl.value;
});
document.getElementById("options").addEventListener("change", function(event) {
    var shouldSave = ["usesidebar", "allman", "leftptr", "globals", "nocasts", "noandor", "noifs", "dark", "regvarsselect", "regvars", "comment_style", "compiler"];
    var options = {};
    for (var key of shouldSave) {
        var el = document.getElementsByName(key)[0];
        options[key] = (el.type === "checkbox" ?
            (el.checked ? "yes" : "no") :
            el.value);
    }
    localStorage.mips_to_c_saved_options = JSON.stringify(options);
});

function updateSidebar() {
    document.body.classList.toggle("has-sidebar", sidebarCheckboxEl.checked);
    formEl.setAttribute("target", sidebarCheckboxEl.checked ? "outputframe" : "");
    try {
        document.getElementsByName("outputframe")[0].contentDocument.body.innerHTML = "";
    } catch (e) {}
}
updateSidebar();
sidebarCheckboxEl.addEventListener("change", updateSidebar);
formEl.addEventListener("submit", function() {
    if (!sidebarCheckboxEl.checked) return;
    try {
        document.getElementsByName("outputframe")[0].contentDocument.body.innerHTML = "Decompiling...";
    } catch (e) {}
});
</script>
</form>
</body>
</html>
"""
    )
