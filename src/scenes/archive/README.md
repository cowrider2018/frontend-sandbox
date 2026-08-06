# 封存的場景

這裡的場景**不在**執行中的 app 裡 —— `src/scenes/index.js` 沒有 import 它們。
程式碼完整保留，隨時可以拿回來。

## 目前封存

### `flow.js` — 01 旋度噪聲流場

GPGPU 粒子系統，位置／速度存在 RGBA32F 貼圖裡，每幀兩道全螢幕 pass 積分整個系統。
渲染用 vertex pulling（`gl_VertexID` → texel fetch），沒有任何頂點屬性，
百萬粒子的頂點頻寬是 0 bytes。另含四分之一解析度的三道 pass bloom
（prefilter → 水平模糊 → 垂直模糊），以及一個把游者尾流灌進速度場的耦合。

**拿回來的方法** —— 在 `src/scenes/index.js` 加兩行：

```js
import flow from './archive/flow.js';
export const SCENES = [flow, fluid, march, reaction];
```

**注意**：`core/agent.js` 已經從「脊椎鏈」改成「球體 + 拖尾歷史」。
`flow.js` 讀的是 `agent.nodes` / `agent.radii` / `agent.count`，這三個屬性名稱與語意
（節點 0 是頭、半徑往後遞減）都還相容，所以它會直接渲染成彗星而不是鰻魚 ——
不需要改任何一行就能跑。
