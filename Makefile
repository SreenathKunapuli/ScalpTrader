# ScalpTrader ops — `make <target>`
# Engine runs in the foreground (Ctrl-C to stop); logs tee to /tmp/scalp_runs.

PY := .venv/bin
LOG_DIR := /tmp/scalp_runs
LOG := $(LOG_DIR)/engine_$(shell date +%Y%m%d).log

.PHONY: run-engine status halt reset watch api web test

run-engine:
	@mkdir -p $(LOG_DIR)
	@caffeinate -is $(PY)/scalpctl run 2>&1 | tee -a $(LOG)

status:
	@$(PY)/scalpctl status

halt:
	@$(PY)/scalpctl halt

reset:
	@$(PY)/scalpctl reset

# live view of the interesting events from today's log (run in a 2nd terminal)
watch:
	@tail -f $(LOG) | grep --line-buffered -E 'scalp\.(entry|qty_zero|rejected|telemetry)|order\.|risk\.|kill|stale|screener'

api:
	@mkdir -p $(LOG_DIR)
	@nohup $(PY)/uvicorn api.app.main:app --host 127.0.0.1 --port 8001 > $(LOG_DIR)/api.log 2>&1 & echo "api -> http://localhost:8001"

web:
	@mkdir -p $(LOG_DIR)
	@cd web && nohup npm run dev > $(LOG_DIR)/web.log 2>&1 & echo "dashboard -> http://localhost:3000"

test:
	@$(PY)/python -m pytest engine/tests research/tests -q
