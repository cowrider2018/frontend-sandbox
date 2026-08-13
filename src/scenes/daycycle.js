/* ── scenes/daycycle.js ──────────────────────────────────────────────
   What time it is, and everything that follows from it.

   One number in — the hour — and out come the four things the rest of
   the scene was already reading anyway: which way the light comes from,
   what colour it is, how much of it there is, and how much daylight is
   in the sky. Nothing downstream had to learn that there is such a thing
   as a time of day; the marcher still reads a direction and a tint, the
   grass still reads the same two, and the sky reads one more float.

   That is the whole reason this is affordable. A day/night cycle usually
   arrives as a second lighting model bolted next to the first one, and
   then every surface in the scene has two ways of being lit and a
   parameter deciding between them. Here it is a *producer* for uniforms
   that already existed.

   ── three states, one control ────────────────────────────────────
   The scene already had a light direction — an XY pad — and a time of
   day is a second answer to the same question. Rather than let the two
   fight, the mode says outright which one is in charge:

     固定    the pad, exactly as before. Every URL and every reference
             shot taken before this file existed still means what it did.
     指定時刻 the hour slider owns the light. The pad goes stale, in the
             same sense the reach slider goes stale under the visibility
             master.
     自動循環 the same, and the hour advances by itself.

   Three modes rather than a switch plus a switch, because they are not
   two independent yes/no questions: "is the pad in charge" and "is the
   clock in charge" cannot both be yes, and a pair of switches is a
   control that can be set to a state with no meaning.

   ── what the sun does at night ───────────────────────────────────
   It sets, and the moon rises opposite it. Both are the same arithmetic
   — one arc, read forwards and backwards — so there is no second body to
   keep in step with the first. What swaps at dusk is which of the two
   the one `uLightDir` is pointing at, and the swap is invisible because
   it happens at the moment both are on the horizon and the directional
   term has already fallen to nothing.
   ------------------------------------------------------------------ */

/* ── the arc ──────────────────────────────────────────────────────
   Sunrise at 06:00, noon at 12:00, sunset at 18:00. Not configurable,
   and not because it would be hard: a scene with one meadow in it has no
   latitude and no season, so every number that would come out of those
   is a number nobody could set correctly. */

/** How high the sun gets at noon, in radians. About 66° — high enough
    for a real midday, low enough that the hills still have north faces. */
const MAX_EL = 1.16;

/** How long a whole day takes when it is running itself, in seconds.
    Compiled in rather than exposed: it is the only number here that is
    about watching the scene rather than about the scene, and a control
    for it would be a control whose correct setting is "however long you
    happen to be looking". Two minutes is short enough to see the shadows
    move and long enough that a still frame is not a smear. */
export const DAY_SECONDS = 130;

/** Where the hour control starts. Mid-afternoon, which is the light the
    scene has always had. */
export const HOUR_DEFAULT = 14.2;

/** The three ways the light can be decided. */
export const DAY_MODES = ['fixed', 'hour', 'cycle'];

/** Whether this mode puts the hour in charge of the light. */
export function hourDrives(mode) { return mode === 'hour' || mode === 'cycle'; }

/* ── the colours ──────────────────────────────────────────────────
   Three, and every other colour of light in the scene is between two of
   them. The tint control still says which hue the light leans, and is
   applied on top of these — so "amber at dawn" and "ice at dawn" are
   still different scenes, and neither of them is midday. */

/** Low sun. Not orange for the sake of a sunset: a sun on the horizon is
    seen through a great deal more air, and what that air takes out first
    is the blue. */
const DAWN = [1.00, 0.42, 0.17];
/** High sun, most of the way to white. */
const NOON = [1.00, 0.93, 0.82];
/** The moon, which is sunlight off a grey rock — so, physically, almost
    the same colour as the sun. It reads blue because everything else has
    gone dark and the eye stops trusting its own white balance, and that
    is a real effect and the one worth reproducing. */
const MOON = [0.50, 0.62, 0.92];

/** How bright the moon is next to the sun. Nothing like the true ratio,
    which is about one part in four hundred thousand and would render as
    black. This is the ratio the eye reports after it has adapted, which
    is the only ratio a picture can be made of. */
const MOON_LIGHT = 0.46;

/* The fill light, which at night is nearly all of the light there is.

   DAY is the three numbers that were hardcoded in three shaders before
   there was a time of day, unchanged to the last digit — the fixed-light
   mode uploads exactly these, so every frame taken before this file
   existed still renders the same.

   NIGHT is not merely DAY scaled down. Scaled down is what makes a night
   scene read as an underexposed day: the sky at night is bluer than it
   is by day *relative to what is left*, because the one thing still
   lighting the ground is the sky itself. The ratio between the channels
   is the whole difference, and the absolute level barely matters. */
const AMBIENT_DAY = [0.100, 0.120, 0.160];
const AMBIENT_NIGHT = [0.055, 0.070, 0.118];

const smooth = (a, b, x) => {
  const u = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return u * u * (3 - 2 * u);
};

const mix3 = (a, b, t) => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

/**
 * The sky at a given hour.
 *
 * @param {number} hour 0..24, wrapped
 * @returns {{dir: number[], tint: number[], day: number, sunUp: boolean,
 *            altitude: number}}
 *   `dir` is the unit direction *toward* the light, as every shadow ray
 *   in this scene expects. `tint` is the light's colour already scaled by
 *   its strength, so that it can be dropped straight into the `uTint`
 *   every surface was already multiplying by — which is why nothing
 *   downstream needed changing. `day` is how much daylight is in the
 *   sky, and is the sky's business alone.
 */
export function skyAt(hour) {
  const t = ((hour % 24) + 24) % 24;

  /* One angle for the whole day: 0 at sunrise, π at sunset, and on round
     through the night. Its sine is the sun's height and its own value is
     the sun's bearing, so the arc and the clock are the same number. */
  const theta = ((t - 6) / 12) * Math.PI;
  const s = Math.sin(theta);

  const sunUp = s > 0;
  /* The moon is the same arc read backwards — highest at midnight,
     opposite the sun in bearing. One body's worth of arithmetic for two
     bodies, and they cannot drift apart because there is only one. */
  const el = (sunUp ? s : -s) * MAX_EL;
  const az = theta - Math.PI / 2 + (sunUp ? 0 : Math.PI);

  const dir = [
    Math.cos(el) * Math.sin(az),
    Math.sin(el),
    Math.cos(el) * Math.cos(az),
  ];

  /* How far up the body that is currently lighting the scene has got.
     Everything about the light's colour and strength hangs off this one
     number rather than off the hour, which is what makes dawn and dusk
     come out identical without either being written down. */
  const altitude = Math.abs(s);
  const high = smooth(0.0, 0.50, altitude);

  let tint;
  if (sunUp) {
    /* Dim at the horizon as well as red. A sunset that only changed hue
       reads as a coloured lamp; what makes it read as *low* is that the
       whole field goes dark at the same time. */
    const strength = 0.22 + 0.78 * high;
    tint = mix3(DAWN, NOON, high).map((c) => c * strength);
  } else {
    tint = MOON.map((c) => c * MOON_LIGHT * (0.45 + 0.55 * high));
  }

  /* How much daylight is in the sky. Deliberately wider than the sun's
     own crossing: the sky is still bright for a while after the sun has
     gone, because it is lit by air rather than by line of sight, and a
     sky that went out the instant the sun touched the horizon is the
     single most common way a day/night cycle gives itself away. */
  const day = smooth(-0.16, 0.10, s);

  return {
    dir,
    tint,
    day,
    sunUp,
    altitude,
    /* Follows the sky rather than the sun, and that is the point: the
       fill is light off the air, so it goes when the air stops being
       lit — not when the sun crosses the horizon. */
    ambient: mix3(AMBIENT_NIGHT, AMBIENT_DAY, day),
  };
}

/** The fill in fixed-light mode: the value the scene has always had. */
export const AMBIENT_FIXED = AMBIENT_DAY;

/**
 * How far the clock has moved the hour on.
 *
 * Kept here rather than in the scene so that "how long a day is" and
 * "what an hour looks like" cannot end up in two files.
 */
export function advanceHour(hour, dt) {
  return (hour + (dt / DAY_SECONDS) * 24) % 24;
}
