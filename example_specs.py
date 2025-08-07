specs = {
    "parsed_request": {
        "original_query": "Create a tool to get my open GitHub PRs and link any relevant issues.",
        "tool_name": "get_open_github_prs_and_link_relevant_issues",
        "description": "Get open GitHub PRs and link any relevant issues",
        "required_unit_functions": [
            "get_current_username",
            "fetch_open_prs",
            "extract_linked_issues"
        ],
        "external_apis_needed": [
            {"github": {"api_token": "GITHUB_TOKEN"}}
        ]
    },
    "full_specs": """
The tool authenticates with GitHub using a token, identifies the current user, fetches all open PRs authored by them (handling pagination), extracts mentioned issues from each PR's body (using patterns like "closes #\d+", "fixes #\d+", "resolves #\d+"), and generates full URLs for those issues in the same repo. Handles errors like invalid tokens or network issues. No external libs beyond `requests` and `re`.
**Unit Specs** (per function):
1. **get_current_username(token)**: Takes token (str); returns username (str). Uses GET /user; requires auth.
2. **fetch_open_prs(token, username)**: Takes token (str), username (str); returns list of dicts [{title: str, url: str, repo: str, body: str}]. Uses GET /search/issues with q="is:pr is:open author:{username}", per_page=100, paginates. Requires auth for rate limits.
3. **extract_linked_issues(pr_body, repo)**: Takes body (str), repo (str, e.g., "owner/repo"); returns list of issue URLs (str). Parses body with regex; builds URLs like "https://github.com/{repo}/issues/{num}".
"""
}

USER_QUERY = 'Create a tool to get my open GitHub PRs and link any relevant issues.'


SPECS_PROMPT = """
You are an expert at designing Python-based tools that integrate with external APIs, based on user queries and provided API documentation. Your task is to analyze the user's query, determine the necessary steps to fulfill it, break it down into a high-level tool with modular unit functions, and generate specifications in the exact format shown below.

Key guidelines:
- The tool should be implementable in Python using only built-in libraries like 'requests' for HTTP calls and 're' for regex if needed. Do not require any external libraries beyond these.
- Identify required external APIs and any authentication tokens needed (e.g., API keys or tokens as environment variables).
- Break the tool into 2-5 simple, reusable unit functions that chain together to achieve the overall goal.
- Handle common edge cases like pagination in API responses, authentication errors, network issues, or parsing failures.
- For any data extraction (e.g., from text bodies), use simple regex patterns based on the docs.
- Function names should be in snake_case.
- Tool name should be a snake_case version of the query's intent.
- Description should be a concise summary.
- In 'full_specs', provide an overall tool description first, then '**Unit Specs** (per function):' followed by numbered specs for each function, including inputs (types), outputs (types), and a brief description of what it does and how (referencing relevant API endpoints or patterns from the docs).

Input:
- Documentation: The following retrieved docs from API references:
{DOCS}

- User Query: {QUERY}

Output the specs exactly in this Python dict format (as a string that could be eval'd), with no additional text:

specs = {{
    "parsed_request": {{
        "original_query": "the user's query here",
        "tool_name": "snake_case_tool_name",
        "description": "Short description of the tool",
        "required_unit_functions": [
            "function1",
            "function2"
        ],
        "external_apis_needed": [
            {{"api_name": {{"key": "ENV_VAR_NAME"}}}}
        ]
    }},
    "full_specs": "
Overall tool description here, including how functions chain, error handling, and constraints.

**Unit Specs** (per function):
1. **function1(inputs)**: Description. Takes type; returns type. Details on implementation.
2. **function2(inputs)**: ...
}}
"
"""

EXAMPLE_DOCS_RETRIEVAL_SPECS = """
== GET the Authenticated User ==

Endpoint Description: Use the REST API to get public and private information about authenticated users.

HTTP Method and Path: get /user

Authentication Requirements: OAuth app tokens and personal access tokens (classic) need the `user` scope in order for the response to include private profile information. The fine-grained token does not require any permissions.

Parameters: None specified.

Response Fields: 
{
  "login": "octocat",
  "id": 1,
  "node_id": "MDQ6VXNlcjE=",
  "avatar_url": "https://github.com/images/error/octocat_happy.gif",
  "gravatar_id": "",
  "url": "https://api.github.com/users/octocat",
  "html_url": "https://github.com/octocat",
  "followers_url": "https://api.github.com/users/octocat/followers",
  "following_url": "https://api.github.com/users/octocat/following{/other_user}",
  "gists_url": "https://api.github.com/users/octocat/gists{/gist_id}",
  "starred_url": "https://api.github.com/users/octocat/starred{/owner}{/repo}",
  "subscriptions_url": "https://api.github.com/users/octocat/subscriptions",
  "organizations_url": "https://api.github.com/users/octocat/orgs",
  "repos_url": "https://api.github.com/users/octocat/repos",
  "events_url": "https://api.github.com/users/octocat/events{/privacy}",
  "received_events_url": "https://api.github.com/users/octocat/received_events",
  "type": "User",
  "site_admin": false,
  "name": "monalisa octocat",
  "company": "GitHub",
  "blog": "https://github.com/blog",
  "location": "San Francisco",
  "email": "octocat@github.com",
  "hireable": false,
  "bio": "There once was...",
  "twitter_username": "monatheoctocat",
  "public_repos": 2,
  "public_gists": 1,
  "followers": 20,
  "following": 0,
  "created_at": "2008-01-14T04:33:35Z",
  "updated_at": "2008-01-14T04:33:35Z",
  "private_gists": 81,
  "total_private_repos": 100,
  "owned_private_repos": 100,
  "disk_usage": 10000,
  "collaborators": 8,
  "two_factor_authentication": true,
  "plan": {
    "name": "Medium",
    "space": 400,
    "private_repos": 20,
    "collaborators": 0
  }
}

Example Request: 
curl -L \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/user

Example Response: 
{
  "login": "octocat",
  "id": 1,
  "node_id": "MDQ6VXNlcjE=",
  "avatar_url": "https://github.com/images/error/octocat_happy.gif",
  "gravatar_id": "",
  "url": "https://api.github.com/users/octocat",
  "html_url": "https://github.com/octocat",
  "followers_url": "https://api.github.com/users/octocat/followers",
  "following_url": "https://api.github.com/users/octocat/following{/other_user}",
  "gists_url": "https://api.github.com/users/octocat/gists{/gist_id}",
  "starred_url": "https://api.github.com/users/octocat/starred{/owner}{/repo}",
  "subscriptions_url": "https://api.github.com/users/octocat/subscriptions",
  "organizations_url": "https://api.github.com/users/octocat/orgs",
  "repos_url": "https://api.github.com/users/octocat/repos",
  "events_url": "https://api.github.com/users/octocat/events{/privacy}",
  "received_events_url": "https://api.github.com/users/octocat/received_events",
  "type": "User",
  "site_admin": false,
}

(Source: https://docs.github.com/en/rest/users/users?apiVersion=2022-11-28#get-the-authenticated-user)

== Search Issues and Pull Requests ==

Endpoint Description: Searches for issues and pull requests. This method returns up to 100 results [per page](https://docs.github.com/rest/guides/using-pagination-in-the-rest-api).

HTTP Method and Path: get /search/issues

Authentication Requirements: This endpoint works with the following fine-grained token types:  
The fine-grained token does not require any permissions.  
This endpoint can be used without authentication if only public resources are requested.

Parameters:
Name, Type, Description  
`accept` string  
Setting to `application/vnd.github+json` is recommended.  

Name, Type, Description  
`q` string Required  
The query contains one or more search keywords and qualifiers. Qualifiers allow you to limit your search to specific areas of GitHub. The REST API supports the same qualifiers as the web interface for GitHub. To learn more about the format of the query, see [Constructing a search query](https://docs.github.com/rest/search/search#constructing-a-search-query). See " [Searching issues and pull requests](https://docs.github.com/search-github/searching-on-github/searching-issues-and-pull-requests)" for a detailed list of qualifiers.  
`sort` string  
Sorts the results of your query by the number of `comments`, `reactions`, `reactions-+1`, `reactions--1`, `reactions-smile`, `reactions-thinking_face`, `reactions-heart`, `reactions-tada`, or `interactions`. You can also sort results by how recently the items were `created` or `updated`, Default: [best match](https://docs.github.com/rest/search/search#ranking-search-results)  
Can be one of: `comments`, `reactions`, `reactions-+1`, `reactions--1`, `reactions-smile`, `reactions-thinking_face`, `reactions-heart`, `reactions-tada`, `interactions`, `created`, `updated`  
`order` string  
Determines whether the first search result returned is the highest number of matches (`desc`) or lowest number of matches (`asc`). This parameter is ignored unless you provide `sort`.  
Default: `desc`  
Can be one of: `desc`, `asc`  
`per_page` integer  
The number of results per page (max 100). For more information, see " [Using pagination in the REST API](https://docs.github.com/rest/using-the-rest-api/using-pagination-in-the-rest-api)."  
Default: `30`  
`page` integer  
The page number of the results to fetch. For more information, see " [Using pagination in the REST API](https://docs.github.com/rest/using-the-rest-api/using-pagination-in-the-rest-api)."  
Default: `1`  
`advanced_search` string  
Set to `true` to use advanced search. Example: `http://api.github.com/search/issues?q={query}&advanced_search=true`

Details on 'q' query parameter and how to use it for PRs, authors, states:  
The query contains one or more search keywords and qualifiers. Qualifiers allow you to limit your search to specific areas of GitHub. The REST API supports the same qualifiers as the web interface for GitHub. To learn more about the format of the query, see [Constructing a search query](https://docs.github.com/rest/search/search#constructing-a-search-query). See " [Searching issues and pull requests](https://docs.github.com/search-github/searching-on-github/searching-issues-and-pull-requests)" for a detailed list of qualifiers.  
Examples from the content include queries like `windows+label:bug+language:python+state:open`, but specific examples for PRs, authors, and states (e.g., 'is:pr is:open author:username') are not explicitly detailed in the provided content under this section. However, based on the reference to searching issues and pull requests and the mention of qualifiers, it aligns with standard GitHub search syntax, which includes:  
- For PRs: Use `is:pr` (e.g., `is:pr is:open author:username` to search for open pull requests by a specific author).  
- For authors: Use `author:username` to filter by the author of the issue or PR.  
- For states: Use `state:open`, `state:closed`, etc., to filter by the state of the issue or PR.

Pagination Details: The number of results per page (max 100). For more information, see " [Using pagination in the REST API](https://docs.github.com/rest/using-the-rest-api/using-pagination-in-the-rest-api)."

(Source: https://docs.github.com/en/rest/search/search?apiVersion=2022-11-28#search-issues-and-pull-requests)

== Linking a Pull Request to an Issue ==

[Linking a pull request to an issue using a keyword]

You can link a pull request to an issue by using a supported keyword in the pull request's description or in a commit message. The pull request must be on the default branch.

* `close`
* `closes`
* `closed`
* `fix`
* `fixes`
* `fixed`
* `resolve`
* `resolves`
* `resolved`

If you use a keyword to reference a pull request comment in another pull request, the pull requests will be linked. Merging the referencing pull request also closes the referenced pull request.

The syntax for closing keywords depends on whether the issue is in the same repository as the pull request.

| Linked issue | Syntax | Example |
|  --  |  --  |  --  |
| Issue in the same repository | KEYWORD #ISSUE-NUMBER | `Closes #10` |
| Issue in a different repository | KEYWORD OWNER/REPOSITORY#ISSUE-NUMBER | `Fixes octo-org/octo-repo#100` |
| Multiple issues | Use full syntax for each issue | `Resolves #10, resolves #123, resolves octo-org/octo-repo#100` |

The keywords can be followed by colons or in uppercase. For example: `Closes: #10`, `CLOSES #10`, or `CLOSES: #10`.

Only manually linked pull requests can be manually unlinked. To unlink an issue that you linked using a keyword, you must edit the pull request description to remove the keyword.

You can also use closing keywords in a commit message. The issue will be closed when you merge the commit into the default branch, but the pull request that contains the commit will not be listed as a linked pull request.

[About linked issues and pull requests]

When you merge a linked pull request into the default branch of a repository, its linked issue is automatically closed. For more information about the default branch, see [Changing the default branch](/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/changing-the-default-branch).

Note

 The special keywords in a pull request description are interpreted only when the pull request targets the repository's default branch. If the pull request targets any other branch, then these keywords are ignored, no links are created, and merging the PR has no effect on the issues.

(Source: https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue)
"""