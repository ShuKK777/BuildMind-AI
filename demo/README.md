# BuildMind AI — 离线演示 / Offline demo

## 启动

先在演示电脑准备 **Node.js 18+** 和现代 Chrome / Edge 浏览器，然后完整下载/解压本目录。运行时无需互联网、API key、npm install 或构建。

- Windows：双击 `start.bat`。
- macOS：终端执行 `bash start.command`；已允许执行时也可双击。
- 任意系统：在本目录运行 `node server.mjs`。
- 浏览器地址：`http://127.0.0.1:4173`。保持启动窗口打开，结束时按 Ctrl+C。
- 不要直接双击 `index.html`：ES module 和 PDF worker 需要本机 HTTP 服务。
- 启动脚本也支持已安装的 Python 3；推荐 Node 路径。端口被占用时先关闭另一个演示窗口，或以 `PORT=4174 node server.mjs` 启动（Windows PowerShell：`$env:PORT=4174; node server.mjs`）。

**English:** Install Node.js 18+ and a modern browser before going offline. Run `node server.mjs` from this directory, then open `http://127.0.0.1:4173`. Everything is bundled locally; there is no build or runtime package installation.

## 推荐演示顺序

1. 智策：查看原始打印街区，拖动旋转、滚轮缩放；点击建筑或用右侧下拉菜单选择。时间线切换建设前、一期、建设后。
2. 点击「替换最高楼 A02」进入智绘。选择潮汐之环、湾区双子或云上花园，修改高度/材质、切换夜景，再应用到街区。只有 A02 被替换；「恢复打印原塔」恢复原始 STL 形态。撤销保留最近五步方案修改。
3. 智管：点击示例问题，查看高亮原文与对应依据；也可上传 DOCX/PDF 体验本地预览。最大 20 MB。旧版 `.doc` 需先另存为 `.docx`。
4. 顶部「一键演示」播放 45 秒/9 阶段的自动场景与字幕。可暂停、继续、重新播放、退出；Esc 退出。手动改方案会结束自动播放。切到其他浏览器标签页时自动暂停。镜头平滑移动，方案与开发阶段淡入淡出，昼夜光照渐变；系统开启“减少动态效果”时自动收敛动画。
5. 点击 EN/中文切换界面与演示字幕；不翻译上传的文档正文。

## 已实现与边界

| 模块 | 当前能力 | 当前边界 |
| --- | --- | --- |
| 智策 | 实际打印零件几何与布局、三维操作、阶段回放 | 地形为简化重建，一期及人口/绿地数值为预设；不是北都官方系统或真实规划分析 |
| 智绘 | 三套本地程序化模型、A02 定点替换、参数调整、撤销恢复 | 没有模型推理或任意参考图生成；不输出可制造 STL，也未验证新方案打印可行性 |
| 智管 | DOCX 文本提取、PDF 分页渲染、示范问题与原文关联 | DOCX 不保留排版/图片；不做 OCR；上传文件不生成审查结果；示范依据不是法定规范 |
| 播放 | 45 秒场景、相机自动环绕、中英字幕 | 不输出视频文件，没有语音解说 |

模型高度以 **打印模型毫米** 标注，不是实际建筑米制高度。A02 零件 66 mm、底标高 6 mm，原塔装配高度 72 mm。

文件在浏览器内处理，不上传到服务器；上传文档只存内存，刷新即丢失。语言与方案选择存入浏览器本地存储。服务仅监听 `127.0.0.1`，没有遥测、远程字体、地图瓦片或 CDN。密码保护/损坏的文档会显示导入错误。

图形优先使用 WebGL2。没有 WebGL 时自动使用共享几何与相机的 CPU 三维兼容渲染，仍支持旋转与选取；兼容模式以最高 30 帧/秒调度并降低栅格分辨率，实际帧率取决于设备；光照/材质效果与 GPU 模式不同。建议正式演示前在目标电脑启用硬件加速并试播。

## 结构

- `app.js`：页面、双语文案、交互、文档状态和演示控制。
- `state.js`：状态校验、预设方案和 45 秒脚本。
- `motion.js`：与帧率无关的演示时钟、暂停恢复与镜头缓动。
- `scene.js` / `print-scene.js`：三维场景、原塔/变体互换、原模型组装。
- `software-renderer.js`：无 WebGL 环境的 CPU 深度缓冲渲染。
- `documents.js`：本地解析入口、示范文档与问题。
- `assets/`：原 STL 导出的紧凑几何和分件清单。
- `vendor/`：固定版本 Three.js、Mammoth、PDF.js 及本地 worker/字体/CMap/WASM。
- `server.mjs`：零 npm 依赖的本机静态服务。

## 验证与维护

```bash
node --test tests/*.test.mjs
```

测试覆盖状态校验、方案参数、演示步骤、示例证据关联、上传边界、打印零件数量/A02 最高及尺寸、资源完整性与本地 HTTP 服务。浏览器验证记录见 [docs/QA.md](docs/QA.md)。

更新依赖仅用于开发：在 `tools/` 运行 `npm ci --omit=optional --ignore-scripts`，然后 `node vendor.mjs`。版本锁定于 `tools/package-lock.json`。用户运行演示不执行这些步骤。

打印几何重建工具：`python tools/convert-print-pack.py /path/to/print-pack.zip`（需要 NumPy；用法见脚本参数）。原上传文件不随演示分发，导出来源与范围见 [docs/PRINT_MODEL.md](docs/PRINT_MODEL.md)。
