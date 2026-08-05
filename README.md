# AETHER — 即時運算視覺實驗室

四座完全跑在 GPU 上的場景。**零依賴、零建置、零打包器**：沒有 npm、沒有框架、沒有函式庫，
`index.html` 雙擊就開。所有東西都是 `<canvas>`、原生 ES modules，與手寫 GLSL。

```
index.html          ← 直接開這個
styles/             ← reset / tokens / shell / ui
src/
  main.js           ← 應用外殼：一個 context、一個 rAF loop、一個 pointer、一個 router
  core/             ← 引擎（與場景無關）
    gl.js           WebGL2 context、材質、render target、全螢幕三角形
    program.js      著色器程式 + uniform 反射
    clock.js        單一 rAF loop，模擬時間與真實時間分離
    pointer.js      滑鼠／觸控／筆統一輸入
    router.js       hash 路由，參數即 URL
    signal.js       ~70 行細粒度反應式系統
    perf.js         GPU timer query + FPS 環形緩衝
  ui/
    widgets.js      自訂元素（slider / switch / segmented / xy-pad）
    panel.js        schema → 控制面板
    chrome.js       tabs、效能 HUD、toast
    cmdk.js         指令面板（模糊搜尋）
  shaders/common.js GLSL 共用區塊（simplex、curl、色彩、tonemap）
  scenes/           四個場景，各自宣告參數 schema
tools/verify.mjs    ← 零依賴的無頭驗收工具（見下）
```

---

## 四個場景

| # | 場景 | 核心技術 |
|---|------|----------|
| 01 | **旋度噪聲流場** | GPGPU：位置／速度存在 RGBA32F 貼圖，每幀兩道全螢幕 pass 積分整個系統。渲染用 vertex pulling（`gl_VertexID` → texel fetch），**沒有任何頂點屬性**，百萬粒子的頂點頻寬是 0 bytes。四分之一解析度 bloom。 |
| 02 | **Navier–Stokes 流體** | Stam 的 stable fluids：半拉格朗日反向追蹤對流（任意 dt 都無條件穩定）→ 渦度強化 → Jacobi 迭代解壓力泊松方程 → 投影成無散度場。每幀 8–40 道 pass。 |
| 03 | **SDF 光線行進** | 整個 3D 場景 **0 頂點 0 三角形**：一道 fragment shader 用 sphere tracing 走進隱式曲面。smooth-min 融合、IQ 軟陰影、5 取樣環境遮蔽、一次鏡面反射彈跳、jitter + 時間累積的免費 TAA。 |
| 04 | **Gray–Scott 反應擴散** | 兩行偏微分方程長出珊瑚、迷宮、細胞分裂與孤立子。九點拉普拉斯算子、每幀 16 個子步、環面邊界（`REPEAT` wrap，所以圖樣可無縫平舖）。 |

---

## 值得一看的幾個決定

**Uniform 反射。** GPU 在連結程式時就已經知道每個 uniform 的名字、型別與位置，所以要求作者
在 JS 裡再抄一遍純粹是重複。[`core/program.js`](src/core/program.js) 讀一次反射資料，
之後 `prog.use({ ... })` 依型別自動分派到正確的 `gl.uniform*`，並把材質綁到自動指派的
texture unit。整個專案沒有一行 `getUniformLocation`。

**沒有建置步驟，所以 GLSL 就是 JS 模組。** 沒有 `#include`，共用著色器碼就放在
[`shaders/common.js`](src/shaders/common.js) 的字串常數裡用 `${}` 組合。這不是妥協 ——
它意味著著色器由瀏覽器自己的模組圖管理，沒有前處理器需要除錯。

**一個超大三角形，不是兩個。** 覆蓋全螢幕用 `(-1,-1) (3,-1) (-1,3)`：沒有對角接縫、少一個頂點，
而且 GPU 不會在斜邊重複光柵化同一個 pixel。

**Ping-pong 是型別的一部分。** `DoubleTarget` 把 `.read` / `.write` / `.swap()` 包在一個物件裡，
消滅了 GPGPU 最常見的 bug：忘記自己在哪一半。

**參數即 URL。** `#/fluid?curl=28&dyeDiss=0.6` 是你眼前畫面的完整描述。拖曳 slider 時用
`replaceState` 寫回位址列 —— 值得分享的狀態就是能寄給別人的狀態。

**自訂元素包住原生控制項。** 每個 widget 底下都是真正的 `<input type=range>` /
`<button role=switch>`，只有像素是我們的。鍵盤操作、螢幕閱讀器語意、IME 行為全部來自平台。

**時間分兩種。** `clock.dt` 是模擬時間（可暫停、可縮放），`clock.wallDt` 是真實時間（永不為 0）。
暫停場景不會讓 UI 動畫卡住；一個異常長的幀（切回分頁、著色器重編譯）永遠不會被交給
積分器 —— dt 上限鎖在 1/20 秒。

---

## 快捷鍵

| 鍵 | 動作 |
|---|---|
`1`–`4` | 切換場景
`Ctrl`+`K` | 指令面板（模糊搜尋所有場景與動作）
`Space` | 暫停 / 繼續
`.` | 暫停時前進一幀
`R` | 重設場景
`S` | 存成 PNG
`P` | 顯示 / 隱藏參數面板
`Z` | 禪模式（隱藏所有介面）
`F` | 全螢幕
`?` | 快捷鍵一覽

畫布上拖曳可互動（攪動流體 / 繞行鏡頭 / 播種化學物質），滾輪縮放。

---

## 需求

WebGL2 + `EXT_color_buffer_float`。每個場景都把模擬狀態存在浮點貼圖裡，沒有浮點 render target
無法運作 —— 缺少時會顯示明確的說明頁，而不是一片黑。

沒有伺服器也能跑嗎？ES modules 受 CORS 限制，所以 `file://` 直接開會被瀏覽器擋。用任何靜態伺服器：

```bash
node tools/verify.mjs --serve     # 內建的，零依賴
# 或
python -m http.server
```

---

## 驗收工具：`tools/verify.mjs`

同樣零依賴。它用 Node 內建的 `WebSocket` 與 `fetch` 直接講 Chrome DevTools Protocol ——
沒有 puppeteer、沒有 playwright、沒有測試框架。協定就是 socket 上的 JSON，這就是全部的依賴表面。

```bash
node tools/verify.mjs                # 每個場景載入、渲染、截圖，回報任何 console error
node tools/verify.mjs --interact     # 端對端互動驗收（真的去拖曳滑鼠、按鍵）
node tools/verify.mjs --responsive   # 手機 / 平板 / 筆電 / 超寬視窗版面與像素預算
node tools/verify.mjs --head         # 開可見的瀏覽器視窗
node tools/verify.mjs --serve        # 只開靜態伺服器
node tools/verify.mjs --eval probe.js --at "#/flow"   # 在頁面裡執行探針，檢查活的 GPU 狀態
```

截圖輸出在 `tools/shots/`。除了抓圖之外它還會斷言：畫布不是單一顏色（全黑代表沒渲染）、
`gl.getError()` 乾淨、沒有未捕捉的例外。

`--interact` 的 17 項檢查涵蓋：intro 關閉、指標事件真的進到畫布、拖曳注入染料、`Ctrl`+`K`
開啟面板、模糊搜尋過濾、面板導航、拖曳繞行鏡頭、數字鍵切換場景、參數往返 URL、
暫停真的停住時鐘、`.` 前進一幀、禪模式、快捷鍵表、tab 點擊、指示器追蹤，以及全程零 console error。

### 它抓到的三個真 bug

1. **`[hidden]` 被 CSS 打敗。** `.fallback { display: grid }` 的優先權高於 UA 對 `[hidden]` 的
   `display: none`，所以「需要 WebGL2」的錯誤畫面一直蓋在正常運作的網站上面。
2. **切換場景時 render target 停在 2×2。** 換場景不會改變畫布尺寸，所以 `_resize()` 的
   「有變化嗎？」提早 return，新場景永遠拿不到真正的解析度。
3. **`gl_PointSize < 1` 的粒子被硬體丟棄。** `ALIASED_POINT_SIZE_RANGE` 在這張 GPU 上從 1 開始，
   而透視除法把點的大小算到 0.29 —— 26 萬個粒子一個都沒畫出來。

第 4 個由 `--interact` 抓到：用 `setPointerCapture` 之後監聽 `pointerleave` 來結束拖曳是錯的，
因為呼叫 capture 本身就會觸發一組 out/leave，拖曳在開始的那一幀就被取消。
