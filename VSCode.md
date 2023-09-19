- [1. 奇技淫巧](#1-奇技淫巧)
  - [a. 调试相关](#a-调试相关)
  - [b. 安装及更新](#b-安装及更新)
  - [c. 设备间同步（VSCode已自带）](#c-设备间同步vscode已自带)
  - [d. 操作技巧](#d-操作技巧)
- [2. LaTeX配置](#2-latex配置)

# 1. 奇技淫巧

## a. 调试相关

* 调试shell：Bash Debug，设置`launch.json`文件，其中`args`需要手动输入，仅用于学习shell，调试则大可不必。
* **Launch**和**Attach**两大功能：后者用于Debug已经在运行的程序
* 条件断点（包括表达式和计数）、LogPoint断点（用于调试无法中断的代码）
* `launch.json`配置

```json
"justMyCode": false,
"stopOnEntry": false,
"cwd": "${workspaceFolder}/examples/midata/v1", //设置python调试路径
"justMyCode": false,
"env": {
    "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/src",  // 可以添加多个PYTHONPATH，`:/;`系统分隔符
    "RANK": "0",
    "WORLD_SIZE": "1",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29500",
    "CUDA_VISIBLE_DEVICES": "7",
    // "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    "PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT": "2"     //调试警告：`pydevd warnings: ... was slow (took 0.8s)`
},
"envFile": "${workspaceFolder}/.env",
```

* `settings.json`配置

```json
"files.autoGuessEncoding": true,
"python.analysis.extraPaths": [],    //pylance不能解析代码时，优先采用 `pyrightconfig.json>>extraPaths`的路径
"python.analysis.logLevel": "Trace",
"remote.SSH.useLocalServer": true,  //remote连接超时，`Resolver error: Error: Connecting with SSH timed out`；设置之后用跳板机，不用重复输入密码
```

* `pyrightconfig.json`配置

```json
"include": ["wespeaker", "tools", "examples/*/*/local"],
"exclude": ["**/data/*", "**/__pycache__", "**/exp/*"],     //用于排除Pylance分析目录
"ignore": [
    "examples/midata/v1/local/timbrelib_expand.py"
],
"extraPaths": [],     //等同`settings.json>>python.analysis.extraPaths`
"defineConstant": {"DEBUG": true},    //可在代码中使用
"typeCheckingMode": "off",  //pyright默认值"basic"，Pylance默认值"off"
"venv": "mispkr",
"venvPath": "/home/xing.chen/usr/local/anaconda3/envs",

"reportMissingImports": true,
"reportMissingTypeStubs": false,
"pythonVersion": "3.9",
"pythonPlatform": "Linux",

"executionEnvironments": []   //用于不同目录下，需要配置不同的环境
```

* `pyrightconfig.json`说明
  + 需要放在VSCode打开文件夹根目录下，[见详细配置说明](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)
  + `Pylance`本质上是使用`pyright`做静态类型检查，[但两者的默认参数可能不一致](https://github.com/microsoft/pylance-release/issues/262)。此处的`pyright`可以通过`pip`安装，并在命令行运行
* 代码提示特别耗时：e.g. `Pylance`卡在某个文件，`1 file to analyse`
  + 工作目录下有特别大的目录，比如`exp, data`等，可以通过`pyrightconfig.json>>exclude`解决
  + 代码问题，比如`embeds = np.stack([kaldiio.load_mat(p) for p in path_list])`；建议`kaldiio.load_mat`这种不要用单行模式写，分开写`Pylance`不会卡死

## b. 安装及更新

* 国内镜像：用`vscode.cdn.azure.cn`/... 替代`az764295.vo.msecnd.net`/...
* 本地更新后，服务器端打不开的问题？**解决**：服务器离线安装VSCode
  + 备份：`~/.vscode-server/extensions`
  + 下载：`Help->About->Commit`，
      - `https://update.code.visualstudio.com/commit:${self_commit}$/server-linux-x64/stable`
      - `https://az764295.vo.msecnd.net/stable/${self_commit}$/vscode-server-linux-x64.tar.gz`
  + 安装：将下载的`vscode-server-linux-x64.tar.gz`上传服务器，解压`tar zxvf ${file_name}$`，移动到`~/.vscode-server/bin`（没有可以自己新建），修改名字为`${self_commit}$`
  + 插件：将备份的插件再移回原位置

## c. 设备间同步（VSCode已自带）

* `VSCode Setting Sync`
  + Gist ID（用于下载设置）：`4378f3fdb8cada30d65d2cee76112c1c`
  + syncLocalSettings.json -> token: `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`

## d. 操作技巧

* `Ctrl+P`：快速选择文件、`>`运行命令、`@/#`在`当前/全局`文件快速定位代码、`:`行数（`Ctrl+G`）
* `Ctrl+Shift+.`：利用当前文件Outline快速定位代码
* `Ctrl + <-/->`按单词跳动光标，`Ctrl`上下箭头可以移动页面
* `Shift + <-/->`选中上下左右内容，`Ctrl+D`匹配最近单词
* `Alt+Shift`上下箭头可以复制，`Alt`上下箭头可以交换行

# 2. LaTeX配置
* 参考资料
  + [LaTeX方案实现](https://juejin.cn/post/6844904061469196302#%E9%85%8D%E7%BD%AE%E6%AD%A3%E5%90%91%E6%90%9C%E7%B4%A2latex-pdf)
  + [正反向搜索配置失效？](https://zhuanlan.zhihu.com/p/434142338)
* `json配置代码如下`

```json
// LaTeX settings start
"latex-workshop.latex.autoBuild.run": "never",
"latex-workshop.message.error.show": true,
"latex-workshop.message.warning.show": false,
"latex-workshop.hover.preview.enabled":true,
"latex-workshop.hover.preview.scale":1.5,
// "latex-workshop.latex.recipe.default": "lastUsed",
"latex-workshop.latex.tools": [
    {
        "name": "xelatex",
        "command": "xelatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOCFILE%"
        ]
    },
    {
        "name": "pdflatex",
        "command": "pdflatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOCFILE%"
        ]
    },
    {
        "name": "bibtex",
        "command": "bibtex",
        "args": [
            "%DOCFILE%"
        ]
    },
    {
        "name": "biber",
        "command": "biber",
        "args": [
            "%DOCFILE%"
        ]
    }
],
"latex-workshop.latex.recipes": [
    {
        "name": "pdflatex",
        "tools": [
            "pdflatex"
        ]
    },
    {
        "name": "pdf->bib->pdf->pdf",
        "tools": [
            "pdflatex",
            "bibtex",
            "pdflatex",
            "pdflatex"
        ]
    },
    {
        "name": "pdf->biber->pdf->pdf",
        "tools": [
            "pdflatex",
            "biber",
            "pdflatex",
            "pdflatex"
        ]
    },
    {
        "name": "xelatex",
        "tools": [
            "xelatex"
        ],
    },
    {
        "name": "xe->bib->xe->xe",
        "tools": [
            "xelatex",
            "bibtex",
            "xelatex",
            "xelatex"
        ]
    },
],
"latex-workshop.latex.clean.fileTypes": [
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.idx",
    "*.ind",
    "*.lof",
    "*.lot",
    "*.out",
    "*.toc",
    "*.acn",
    "*.acr",
    "*.alg",
    "*.glg",
    "*.glo",
    "*.gls",
    "*.ist",
    "*.fls",
    "*.log",
    "*.fdb_latexmk",
    "*.nav",
    "*.snm",
    // "*.synctex.gz",
    // "*.bib",
    "*.run.xml"
],
"latex-workshop.view.pdf.viewer":"external",
// "latex-workshop.view.pdf.ref.viewer":"external",
"latex-workshop.view.pdf.external.viewer.command": "D:/VSCode/SumatraPDF_3.4.4_Install/SumatraPDF/SumatraPDF.exe",
"latex-workshop.view.pdf.external.viewer.args": [
    "--unique",
    "%PDF%",
],
// forward-search
"latex-workshop.view.pdf.external.synctex.command": "D:/VSCode/SumatraPDF_3.4.4_Install/SumatraPDF/SumatraPDF.exe",
"latex-workshop.view.pdf.external.synctex.args": [
    "-forward-search",
    "%TEX%",
    "%LINE%",
    "%PDF%"
],
//backward-search, in SumatraPDF>>Settings>>Options
//WORK -> "D:\VSCode\Microsoft VS Code\Code.exe" -g "%f":"%l"
//LaTeX settings end

```
