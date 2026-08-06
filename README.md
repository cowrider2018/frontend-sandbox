# AETHER — SDF 光線行進

一個完全跑在 GPU 上的場景。**零依賴、零建置、零打包器**：沒有 npm、沒有框架、沒有函式庫，
`index.html` 開起來就是全部。所有東西都是 `<canvas>`、原生 ES modules，與手寫 GLSL。

整個場景**沒有任何一個頂點、沒有任何一個三角形**。一道 fragment shader 沿著射線走進由距離函數
定義的隱式曲面：一團互相以 smooth-min 焊接的公轉球體，加上一個以自己的節奏翻滾的環。
陰影來自朝光源再走一次，環境遮蔽來自在命中點周圍取樣距離場，反射來自沿鏡射方向再走一次，
抗鋸齒來自每幀抖動取樣點再讓時間累積把它收斂掉。

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
  shaders/common.js GLSL 共用區塊（simplex、色彩、tonemap）
  scenes/march.js   場景本體
archive/            ← 不在 app 裡的東西，附完整接回步驟（見該目錄的 README）
tools/verify.mjs    ← 零依賴的無頭驗收工具（見下）
```

---

## 值得一看的幾個決定

**Uniform 反射。** GPU 在連結程式時就已經知道每個 uniform 的名字、型別與位置，所以要求作者
在 JS 裡再抄一遍純粹是重複。[`core/program.js`](src/core/program.js) 讀一次反射資料，
之後 `prog.use({ ... })` 依型別自動分派到正確的 `gl.uniform*`，並把材質綁到自動指派的
texture unit。整個專案沒有一行 `getUniformLocation`。

**沒有建置步驟，所以 GLSL 就是 JS 模組。** 沒有 `#include`，共用著色器碼就放在
[`shaders/common.js`](src/shaders/common.js) 的字串常數裡用 `${}` 組合。這不是妥協 ——
它意味著著色器由瀏覽器自己的模組圖管理，沒有前處理器需要除錯。代價是：GLSL 註解裡不能有反引號，
不然樣板字串會提前結束（這件事花了一次 boot failure 才學會）。

**一個超大三角形，不是兩個。** 覆蓋全螢幕用 `(-1,-1) (3,-1) (-1,3)`：沒有對角接縫、少一個頂點，
而且 GPU 不會在斜邊重複光柵化同一個 pixel。

**Ping-pong 是型別的一部分。** `DoubleTarget` 把 `.read` / `.write` / `.swap()` 包在一個物件裡，
消滅了最常見的 bug：忘記自己在哪一半。時間累積就是靠它 —— 對著同時是當前 colour attachment
的材質取樣是未定義行為，而它產生的瑕疵看起來夠合理，合理到可以浪費你一個晚上。

**免費的抗鋸齒。** 每個 pixel 每幀把射線方向抖動不到一個 pixel，再讓時間累積把這些樣本平均掉。
鏡頭移動時混合係數自動調低，否則抖動會變成拖影。

**參數即 URL。** `#/march?balls=9&blend=0.55` 是你眼前畫面的完整描述。拖曳 slider 時用
`replaceState` 寫回位址列 —— 值得分享的狀態就是能寄給別人的狀態。驗收工具也靠這個擺場景。

**自訂元素包住原生控制項。** 每個 widget 底下都是真正的 `<input type=range>` /
`<button role=switch>`，只有像素是我們的。鍵盤操作、螢幕閱讀器語意、IME 行為全部來自平台。

**時間分兩種。** `clock.dt` 是模擬時間（可暫停、可縮放），`clock.wallDt` 是真實時間（永不為 0）。
暫停場景不會讓 UI 動畫卡住；一個異常長的幀（切回分頁、著色器重編譯）永遠不會被交給
積分器 —— dt 上限鎖在 1/20 秒。

---

## 快捷鍵

| 鍵 | 動作 |
|---|---|
`Ctrl`+`K` | 指令面板（模糊搜尋所有動作）
`Space` | 暫停 / 繼續
`.` | 暫停時前進一幀
`R` | 重設場景
`S` | 存成 PNG
`P` | 顯示 / 隱藏參數面板
`Z` | 禪模式（隱藏所有介面）
`F` | 全螢幕
`?` | 快捷鍵一覽

拖曳畫布繞行鏡頭，滾輪縮放，面板裡的 XY 盤轉動光源。

---

## 需求

WebGL2 + `EXT_color_buffer_float`。時間累積存在浮點貼圖裡，沒有浮點 render target 無法運作 ——
缺少時會顯示明確的說明頁，而不是一片黑。

ES modules 受 CORS 限制，`file://` 直接開會被瀏覽器擋。用任何靜態伺服器：

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
node tools/verify.mjs                # 六種設定各載入、渲染、截圖，回報任何 console error
node tools/verify.mjs --interact     # 端對端互動驗收（真的去拖曳滑鼠、滾滾輪、按鍵）
node tools/verify.mjs --responsive   # 手機 / 平板 / 筆電 / 超寬視窗版面與像素預算
node tools/verify.mjs --head         # 開可見的瀏覽器視窗
node tools/verify.mjs --serve        # 只開靜態伺服器
node tools/verify.mjs --eval probe.js --at "#/march"   # 在頁面裡執行探針，檢查活的 GPU 狀態
```

截圖輸出在 `tools/shots/`。除了抓圖之外它還會斷言：畫布不是單一顏色（全黑代表沒渲染）、
`gl.getError()` 乾淨、沒有未捕捉的例外。

`--interact` 的 17 項檢查涵蓋：intro 關閉、指標事件真的進到畫布、拖曳繞行鏡頭、滾輪縮放、
XY 盤轉動光源、`Ctrl`+`K` 面板、模糊搜尋、渲染縮放真的重配了 render target、參數往返 URL、
暫停真的停住時鐘、`.` 前進一幀、禪模式、快捷鍵表、單場景時隱藏 tab bar、恢復預設值，
以及全程零 console error。

### 它抓到的八個真 bug

1. **`[hidden]` 被 CSS 打敗。** `.fallback { display: grid }` 的優先權高於 UA 對 `[hidden]` 的
   `display: none`，所以「需要 WebGL2」的錯誤畫面一直蓋在正常運作的網站上面。
2. **切換場景時 render target 停在 2×2。** 換場景不會改變畫布尺寸，所以 `_resize()` 的
   「有變化嗎？」提早 return，新場景永遠拿不到真正的解析度。
3. **`gl_PointSize < 1` 的粒子被硬體丟棄。** `ALIASED_POINT_SIZE_RANGE` 在這張 GPU 上從 1 開始，
   而透視除法把點的大小算到 0.29 —— 26 萬個粒子一個都沒畫出來。
4. **拖曳在開始的那一幀就被取消。** 用 `setPointerCapture` 之後又監聽 `pointerleave` 來結束拖曳
   是錯的：呼叫 capture 本身就會觸發一組 out/leave。
5. **場景上下顛倒。** `right` 算成了 `cross(worldUp, fwd)` 而不是 `cross(fwd, worldUp)`，
   負號穿過下一個外積傳給 `up` —— 格線地板一直畫在天空裡，而場景夠對稱，
   對稱到這件事看起來像是刻意的構圖。
6. **View Transition 被中斷時丟出未處理的 rejection。**
7. **靜態的 hint 區塊被當成有值的控制項註冊。** 它帶了 `id`，於是進了 state map，
   第一次程式化 `setValues()` 打到一個沒有 `set()` 的元素就爆掉。
8. **低幀率時點擊沒有反應。** 靠算繪迴圈取樣 `pointer.down` 來判斷點擊是錯的，
   25 fps 下一次快速點擊可以整個發生在兩幀之間 —— 要在 pointer 事件上判定。
   （這個修正留在 `core/pointer.js` 的 `onDown` / `onUp` 回呼裡，目前沒有人用。）

第 5 個特別值得一提：它是被截圖裡「格線出現在地平線上方」這個細節抓到的，
而不是被任何斷言抓到的 —— 這正是為什麼驗收工具要輸出可以用眼睛看的東西。

---

## `archive/`

不在 app 裡、但完整保留並附上接回步驟的東西：**游者**（球體 + 慣性剝離小球、
轉向、碰撞反射、停棲、彈射）與 **01 旋度噪聲流場**。
02 流體與 04 反應擴散只存在於 git 裡，取回指令也寫在那份 README。
