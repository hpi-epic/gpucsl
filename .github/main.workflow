# workflow auto-pr
workflow "auto-pr" {
  resolves = ["create-pr"]
  on = "push"
}

action "create-pr" {
  uses = "smartinspereira/auto-create-pr-action@master"
  secrets = ["GITHUB_TOKEN"]
  env = {
    BRANCH_PREFIX = ""
    # BASE_BRANCH = ""
    # PULL_REQUEST_TITLE = ""
    # PULL_REQUEST_BODY = ""
    PULL_REQUEST_DRAFT = "true"
  }
}
