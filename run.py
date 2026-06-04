from agents.loan_graph import loan_agent

initial_state = {
    "case_id":     "CASE001",   # must match a folder in case_documents/
    "documents":   {},
    "validation":  {},
    "fraud":       {},
    "decision":    {},
    "report_path": "",
    "status":      "new",
    "retry_count": 0,
    "error":       "",
}

result = loan_agent.invoke(initial_state)

print("\n=== FINAL STATUS:", result["status"])
print("=== DECISION:    ", result["decision"].get("recommendation"))
print("=== REPORT PATH: ", result["report_path"])