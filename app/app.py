from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist, EmailStr
from typing import Optional, Dict, List
import os
import httpx

app = FastAPI(title="TRU-e Calculator", version="0.2.1")

# -------------------- CORS --------------------
# Permite localhost, 127.0.0.1, *.app.github.dev (Codespaces) y *.vercel.app
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?:\/\/([a-z0-9-]+-)?(localhost(:\d+)?|127\.0\.0\.1(:\d+)?|.*\.app\.github\.dev|.*\.vercel\.app)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== MODELOS =====================
class ESGSub(BaseModel):
    sa: Optional[float] = None
    ss: Optional[float] = None
    co2: Optional[float] = None

class GOBSub(BaseModel):
    gc: Optional[float] = None
    gri: Optional[float] = None

class Inputs(BaseModel):
    # AOV
    sla: float = 0
    complaints_rate: float = 0
    productivity_per_labor_hour: float = 0
    caov: float = 0
    esg: Optional[float] = None
    governance: Optional[float] = None
    esg_sub: Optional[ESGSub] = None
    gob_sub: Optional[GOBSub] = None
    # DAO / CUP
    nps: float = 0
    satisfaction: float = 0
    digital_rep: float = 0
    brand_promise: float = 0
    brand_perception: float = 0
    # ATRU
    sres: Optional[float] = None
    delta_theta: Optional[float] = None
    # WTP / precio
    wtp_premium_pct: float = 0
    price_reference: Optional[float] = None
    price_real: Optional[float] = None
    currency: Optional[str] = None

class ScoreRequest(BaseModel):
    industry: str = Field(default="_generic")
    brand: str = Field(default="Marca")
    weights_mode: str = Field(default="auto_heuristic")
    weights_override: Optional[Dict[str, float]] = None
    data: Inputs

class ScoreResponse(BaseModel):
    brand: str
    industry: str
    score_0_100: float
    label: str                 # A, B, B+, A, A+, C...
    level: str                 # Funcional..Transformacional
    nm_numeric: int            # 1..5
    label_combined: str        # p.ej. "Relacional – B+ (70–79)"
    c_factor: float            # 0.5 / 1.0 / 1.5
    aov_0_100: float
    dao_0_100: float
    coherence_0_100: float
    atru_0_100: float
    wtp_impact_0_100: float
    wtp_price_gap_abs: Optional[float] = None
    wtp_price_gap_pct: Optional[float] = None
    weights_used: Dict[str, float]
    components: Dict[str, float]

class BrandPayload(BaseModel):
    brand: str
    data: Inputs

class ScoreBatchRequest(BaseModel):
    industry: str
    weights_mode: str = "auto_heuristic"
    weights_override: Optional[Dict[str, float]] = None
    brands: conlist(BrandPayload, min_length=1, max_length=5)

class ScoreBatchResponse(BaseModel):
    industry: str
    results: List[ScoreResponse]

# ===================== UTILIDADES =====================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def n01(x: float) -> float:
    return round(clamp01(x) * 100.0, 2)

def esg_comp(esg: Optional[float], sub: Optional[ESGSub]) -> float:
    if sub and sub.sa is not None and sub.ss is not None and sub.co2 is not None:
        return clamp01((0.4*sub.sa + 0.3*sub.ss + 0.3*sub.co2) / 100.0)
    if esg is not None:
        return clamp01(esg / 100.0)
    return 0.0

def gob_comp(gob: Optional[float], sub: Optional[GOBSub]) -> float:
    if sub and sub.gc is not None and sub.gri is not None:
        return clamp01((0.6*sub.gc + 0.4*sub.gri) / 100.0)
    if gob is not None:
        return clamp01(gob / 100.0)
    return 0.0

def industry_weights(industry: str) -> Dict[str, float]:
    base = {"sla":0.18,"cup":0.22,"complaints":0.18,"caov":0.12,"productivity":0.10,"esg":0.10,"governance":0.10}
    ind = industry.lower()
    if "salud" in ind or "health" in ind:
        return {"sla":0.24,"cup":0.18,"complaints":0.20,"caov":0.14,"productivity":0.06,"esg":0.09,"governance":0.09}
    if "retail" in ind or "comercio" in ind:
        return {"sla":0.16,"cup":0.28,"complaints":0.16,"caov":0.10,"productivity":0.12,"esg":0.09,"governance":0.09}
    if any(k in ind for k in ["bank","finanzas","finance"]):
        return {"sla":0.18,"cup":0.22,"complaints":0.18,"caov":0.14,"productivity":0.10,"esg":0.09,"governance":0.09}
    return base

def cup_from_perception(nps: float, satisfaction: float, digital_rep: float,
                        brand_promise: float, brand_perception: float) -> Dict[str, float]:
    nps01 = clamp01((nps + 100.0) / 200.0)
    sat01 = clamp01(satisfaction / 100.0)
    dig01 = clamp01(digital_rep / 100.0)
    prom01 = clamp01(brand_promise / 100.0)
    perc01 = clamp01(brand_perception / 100.0)
    gap = max(0.0, prom01 - perc01)
    gap_penalty = 1.0 - gap
    perception = 0.4*nps01 + 0.35*sat01 + 0.25*dig01
    cup = clamp01(0.7*perception + 0.3*gap_penalty)
    return {"cup01": cup, "nps01": nps01, "sat01": sat01, "dig01": dig01, "gap_penalty01": gap_penalty}

def aov_block(i: Inputs) -> Dict[str, float]:
    sla01 = clamp01(i.sla / 100.0)
    complaints01 = 1.0 - clamp01(i.complaints_rate / 50.0)
    prod01 = clamp01(i.productivity_per_labor_hour / 200.0)
    esg01 = esg_comp(i.esg, i.esg_sub)
    gob01 = gob_comp(i.governance, i.gob_sub)
    caov01 = clamp01(i.caov / 100.0)
    aov01 = clamp01(0.22*sla01 + 0.18*complaints01 + 0.14*prod01 + 0.16*esg01 + 0.16*gob01 + 0.14*caov01)
    return {"aov01": aov01, "sla01": sla01, "complaints01": complaints01, "prod01": prod01,
            "esg01": esg01, "gob01": gob01, "caov01": caov01}

def coherence_continuous(aov01: float, dao01: float) -> float:
    return clamp01(1.0 - abs(aov01 - dao01))

def c_discrete_from_coherence(coh01: float) -> float:
    if coh01 >= 0.8: return 1.5
    if coh01 >= 0.6: return 1.0
    return 0.5

def atru_formal(aov01: float, dao01: float, sres: Optional[float], delta_theta: Optional[float]) -> float:
    phi = 1.618
    h = clamp01(1.0 - abs(dao01 - aov01))
    s = clamp01(sres) if sres is not None else 0.0
    dt = clamp01(delta_theta) if delta_theta is not None else 0.0
    atru = (1.0 - s) * (phi ** (1.0 - dt)) * h
    return clamp01(atru / 1.7)  # normalización suave

def wtp_gap(price_reference: Optional[float], price_real: Optional[float], wtp_premium_pct: float):
    if price_reference is None or price_real is None or price_real == 0:
        return {"gap_abs": None, "gap_pct": None}
    precio_wtp = (1.0 + wtp_premium_pct) * price_reference
    gap_abs = precio_wtp - price_real
    gap_pct = gap_abs / price_real
    return {"gap_abs": round(gap_abs, 4), "gap_pct": round(gap_pct, 4)}

def label_from_score(s: float) -> str:
    if s >= 90: return "A+"
    if s >= 80: return "A"
    if s >= 70: return "B+"
    if s >= 60: return "B"
    return "C"

def level_from_score(s: float) -> str:
    if s >= 85: return "Transformacional"
    if s >= 70: return "Inspiracional"
    if s >= 55: return "Relacional"
    if s >= 40: return "Operacional"
    return "Funcional"

def nm_from_level(s: float) -> int:
    if s >= 85: return 5
    if s >= 70: return 4
    if s >= 55: return 3
    if s >= 40: return 2
    return 1

def combined_label(level: str, score: float) -> str:
    if score < 60: return f"{level} – C (<60)"
    if score < 70: return f"{level} – B (60–69)"
    if score < 80: return f"{level} – B+ (70–79)"
    if score < 90: return f"{level} – A (80–89)"
    return f"{level} – A+ (90–100)"

def weights_for(industry: str, override: Optional[Dict[str, float]]) -> Dict[str, float]:
    w = override if override else industry_weights(industry)
    total = sum(w.values()) or 1.0
    return {k: v/total for k, v in w.items()}

# ===================== LÓGICA =====================
def compute_one(industry: str, brand: str, inp: Inputs, w_override: Optional[Dict[str, float]]=None) -> ScoreResponse:
    w = weights_for(industry, w_override)

    cup = cup_from_perception(inp.nps, inp.satisfaction, inp.digital_rep, inp.brand_promise, inp.brand_perception)
    dao01 = cup["cup01"]
    aov = aov_block(inp); aov01 = aov["aov01"]

    coh01 = coherence_continuous(aov01, dao01)
    c_factor = c_discrete_from_coherence(coh01)
    atru01 = atru_formal(aov01, dao01, inp.sres, inp.delta_theta)
    wtp_impact01 = clamp01(inp.wtp_premium_pct / 0.3)
    gap = wtp_gap(inp.price_reference, inp.price_real, inp.wtp_premium_pct)

    agg = (
        w["sla"]          * clamp01(inp.sla / 100.0) +
        w["cup"]          * dao01 +
        w["complaints"]   * clamp01(1.0 - inp.complaints_rate / 50.0) +
        w["caov"]         * clamp01(inp.caov / 100.0) +
        w["productivity"] * clamp01(inp.productivity_per_labor_hour / 200.0) +
        w["esg"]          * aov["esg01"] +
        w["governance"]   * aov["gob01"]
    )

    agg = clamp01(0.80*agg + 0.15*atru01 + 0.05*wtp_impact01)
    agg = clamp01(agg * (0.5 + (c_factor - 0.5) * 0.5))

    score_100 = n01(agg)
    label = label_from_score(score_100)
    level = level_from_score(score_100)
    nm = nm_from_level(score_100)
    combo = combined_label(level, score_100)

    return ScoreResponse(
        brand=brand,
        industry=industry,
        score_0_100=score_100,
        label=label,
        level=level,
        nm_numeric=nm,
        label_combined=combo,
        c_factor=c_factor,
        aov_0_100=n01(aov01),
        dao_0_100=n01(dao01),
        coherence_0_100=n01(coh01),
        atru_0_100=n01(atru01),
        wtp_impact_0_100=n01(wtp_impact01),
        wtp_price_gap_abs=gap["gap_abs"],
        wtp_price_gap_pct=gap["gap_pct"],
        weights_used=w,
        components={
            "sla": n01(aov["sla01"]),
            "complaints": n01(aov["complaints01"]),
            "productivity": n01(aov["prod01"]),
            "esg": n01(aov["esg01"]),
            "governance": n01(aov["gob01"]),
            "caov": n01(aov["caov01"]),
            "nps": n01(cup["nps01"]),
            "satisfaction": n01(cup["sat01"]),
            "digital_rep": n01(cup["dig01"]),
            "cup_gap_penalty": n01(cup["gap_penalty01"]),
        }
    )

# ===================== ENDPOINTS =====================
@app.get("/health", tags=["meta"])
def health():
    return {"ok": True, "service": "tru-e-calculator", "version": "0.2.1"}

@app.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest):
    return compute_one(payload.industry, payload.brand, payload.data, payload.weights_override)

@app.post("/score-batch", response_model=ScoreBatchResponse)
def score_batch(payload: ScoreBatchRequest):
    results = [compute_one(payload.industry, b.brand, b.data, payload.weights_override) for b in payload.brands]
    return ScoreBatchResponse(industry=payload.industry, results=results)

@app.get("/", tags=["meta"])
def _root():
    return {"ok": True, "docs": "/docs", "health": "/health"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

# ===================== /lead → Supabase =====================
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE", "")
LEADS_TABLE = "leads"

class LeadIn(BaseModel):
    email: EmailStr
    brand: str
    industry: str
    snapshot: dict
    source: Optional[str] = "quick-check"

@app.post("/lead")
async def lead(payload: LeadIn):
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE:
        raise HTTPException(status_code=500, detail="Supabase credentials missing")

    url = f"{SUPABASE_URL}/rest/v1/{LEADS_TABLE}"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=headers, json=payload.dict(), params={"select": "*"})
        if r.status_code >= 400:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise HTTPException(status_code=r.status_code, detail={"supabase_error": detail})

    return r.json()

