import httpx
# 이 부분을 수정하거나 추가해주세요.
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ----------------------------------------------------
# 1. FastAPI 애플리케이션 생성
# ----------------------------------------------------
app = FastAPI()

# ----------------------------------------------------
# 2. 데이터 모델 정의 (Pydantic BaseModel 사용)
# ----------------------------------------------------
# /ask 엔드포인트로 들어오는 요청의 본문을 위한 모델
class AskRequest(BaseModel):
    question: str
    context: str

# ----------------------------------------------------
# 3. 서버 내부 상태 관리
# ----------------------------------------------------
# 호출할 LLM 모델의 이름을 저장하는 변수
# 서버가 시작될 때 기본값은 'Qwen' 입니다.
CURRENT_MODEL = "Qwen"
BASE_URL = "http://10.23.80.35:8000/v2/models"

# ----------------------------------------------------
# 4. API 엔드포인트 구현
# ----------------------------------------------------

@app.get("/status", summary="서버 상태 확인")
async def get_server_status():
    """
    서버가 정상적으로 실행 중인지 확인하고 현재 설정된 모델 정보를 반환합니다.
    """
    return {"status": "ok", "message": "API 서버가 정상적으로 동작 중입니다.", "current_model": CURRENT_MODEL}

@app.post("/qwen", summary="LLM 모델을 Qwen으로 변경")
async def set_model_to_qwen():
    """
    호출할 LLM 모델을 'Qwen'으로 설정합니다.
    """
    global CURRENT_MODEL
    CURRENT_MODEL = "Qwen"
    return {"message": f"성공적으로 모델을 '{CURRENT_MODEL}'으로 변경했습니다."}

@app.post("/mis", summary="LLM 모델을 MIS로 변경")
async def set_model_to_mis():
    """
    호출할 LLM 모델을 'MIS'로 설정합니다.
    (실제 모델 이름이 다르다면 이 부분을 수정해주세요.)
    """
    global CURRENT_MODEL
    # 'mis'가 어떤 모델을 지칭하는지 몰라 임의로 'MIS'로 지정했습니다.
    # 실제 호출해야 하는 모델 이름으로 변경해주세요.
    CURRENT_MODEL = "MIS"
    return {"message": f"성공적으로 모델을 '{CURRENT_MODEL}'으로 변경했습니다."}


@app.post("/ask", summary="LLM 모델에 질문하고 답변 받기")
async def ask_question(request: AskRequest):
    """
    클라이언트로부터 질문(question)과 문맥(context)을 받아
    설정된 LLM 모델 API에 요청을 보내고, 그 결과를 반환합니다.
    """
    # 1. LLM 모델 API에 보낼 프롬프트 구성
    # 요청하신 형식에 맞춰 question과 context를 조합합니다.
    prompt = f"<|im_start|>system\n{request.context}<|im_end|>\n<|im_start|>user\n{request.question}<|im_end|>\n<|im_start|>assistant"

    # 2. LLM 모델 API에 보낼 요청 본문(payload) 생성
    payload = {
        "inputs": [
            {
                "name": "PROMPT",
                "shape": [1],
                "datatype": "BYTES",
                "data": [prompt]
            }
        ],
        "outputs": [
            {"name": "RESPONSE"}
        ]
    }

    # 3. 비동기 HTTP 클라이언트를 사용하여 LLM API 호출
    target_url = f"{BASE_URL}/{CURRENT_MODEL}/infer"
    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(target_url, json=payload, headers=headers, timeout=30.0)

            # LLM 서버가 에러를 반환하면 우리 서버에서도 에러를 발생시킴
            response.raise_for_status()

            # 4. LLM 서버로부터 받은 응답 파싱
            response_data = response.json()

            # 응답 구조를 기반으로 최종 답변 추출
            # 'outputs' 리스트의 첫 번째 요소의 'data' 리스트의 첫 번째 값을 가져옵니다.
            answer = response_data.get("outputs", [{}])[0].get("data", [""])[0]

            return {"answer": answer, "model_used": CURRENT_MODEL}

    except httpx.RequestError as e:
        # LLM 서버 호출 중 네트워크 관련 에러 발생 시
        raise HTTPException(status_code=503, detail=f"LLM 모델 서버({e.request.url}) 호출에 실패했습니다.")
    except Exception as e:
        # 기타 예외 처리
        raise HTTPException(status_code=500, detail=f"알 수 없는 오류가 발생했습니다: {str(e)}")

    # ----------------------------------------------------
# 5. 웹 프론트엔드 제공 (이 부분이 있는지 확인!)
# ----------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    서버의 루트 경로('/')로 접속하면 index.html 파일을 읽어서 반환합니다.
    """
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)