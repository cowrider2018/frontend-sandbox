# 封存

這裡的東西**不在**執行中的 app 裡 —— 沒有任何一個檔案 import 它們。
程式碼完整保留，而且每一項都寫了怎麼接回去。

---

## `agent.js` — 游者（球體 + 慣性剝離小球）

一顆在「正規空間」（原點附近半徑 1.9 的球）裡活動的球體，加上因加速而從本體剝離、
各自帶著慣性飄開並逐漸消散的小球。

**它自己的內容**（這個檔案就是全部）：

- Reynolds 轉向：算出期望速度然後朝它加速；速度同時有上限與**下限**
- 剝離率由**加速度**驅動而不是速度；固定 10 個槽位的池子，半徑用 `pow(life, 0.45)` 收縮
- 兩個衝量原語：`reflect(nx, ny, nz, bounce)` 與 `displace(dx, dy, dz)`
- `launch(x, y, z, power)` 朝某方向大力扔出
- `frozen` 旗標：場景自己接管位置時（停棲）用
- `flatten` 參數：把它壓向 z=0 平面，給 2D 場景用
- `bodies(out, capacity, scale, offsetY, girth)` / `boundingSphere(...)`：
  打包成 `vec4(x, y, z, radius)` 陣列給 shader

### 接回 app（4 個點）

**1. `src/main.js`** — App 擁有它，換場景時保留動量：

```js
import { Agent } from '../archive/agent.js';        // 或搬回 src/core/

// 建構子
this.agent = new Agent();
this.sceneCtx = { /* … */ agent: this.agent };

// frame() 開頭，scene.frame() 之前
if (this.state.agent !== false) {
  this.agent.update(clock.dt, {
    mode: this.pointer.active ? (this.state.agentMode ?? 'wander') : 'wander',
    speed: this.state.agentSpeed ?? 1.0,
    flatten: this.sceneDef?.agentFlatten ?? 0,
  });
}
```

**2. 指標事件轉發**（點擊彈射需要，不能在算繪迴圈裡取樣 `pointer.down` ——
25 fps 下一次快速點擊可能整個發生在兩幀之間）：

```js
this.pointer = new Pointer(canvas, {
  onDown: (p) => this.scene?.onPointerDown?.(p),
  onUp:   (p) => this.scene?.onPointerUp?.(p),
});
```

**3. 場景參數**：場景要宣告 `agent`（switch）、`agentMode`（select：
`wander` / `follow` / `flee`）、`agentSpeed`（slider）。App 靠這三個共通 id 驅動它，
不需要知道是哪個場景在跑。2D 場景另外在模組上宣告 `agentFlatten: 1`。

**4. 場景耦合**：這是真正的工作量，而且每個場景都不一樣。完整可運作的版本在：

```bash
git show 01c865d:src/scenes/march.js      # 球體進 SDF、碰撞、停棲、彈射、漣漪、輝光 pass
git show efb2553:src/scenes/fluid.js      # 游者當攪動者，沿身體注入速度與染料
git show efb2553:src/scenes/reaction.js   # 頭部切開通道、身體撒下菌落
git show 287d8f4:archive/flow.js          # 尾流灌進速度場 + 發光緞帶（當時在 src/scenes/）
```

`01c865d` 的 `march.js` 特別完整：包含把星體位置搬到 CPU（碰撞需要在 JS 裡知道東西在哪）、
包圍球提前退出、漣漪當成距離場的加項、以及點擊與拖曳的區分。

---

## `flow.js` — 01 旋度噪聲流場

GPGPU 粒子系統，位置／速度存在 RGBA32F 貼圖裡，每幀兩道全螢幕 pass 積分整個系統。
渲染用 vertex pulling（`gl_VertexID` → texel fetch），沒有任何頂點屬性，
百萬粒子的頂點頻寬是 0 bytes。另含四分之一解析度的三道 pass bloom
（prefilter → 水平模糊 → 垂直模糊）。

**接回來**：把檔案移到 `src/scenes/`，然後在 `src/scenes/index.js` 加兩行：

```js
import flow from './flow.js';
export const SCENES = [flow, march];
```

**注意**：它含有游者的耦合（`uSwimmer` / `uWake` uniform 與 `VERT_SWIMMER` pass）。
沒有 agent 的話，把 `state.agent !== false` 的那個區塊拿掉、
並把 `uWake` 傳 0 即可，其餘部分獨立運作。

---

## 其他從 app 移除、只存在於 git 的東西

```bash
git show 979cc95:src/scenes/fluid.js    > src/scenes/fluid.js    # 02 Navier–Stokes 流體
git show 979cc95:src/scenes/reaction.js > src/scenes/reaction.js # 04 Gray–Scott 反應擴散
```

兩者都是自足的場景模組，放回 `src/scenes/` 並加進 `index.js` 就能跑
（`979cc95` 的版本沒有游者耦合，所以不需要 agent）。
