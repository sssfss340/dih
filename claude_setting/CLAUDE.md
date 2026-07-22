# git規範
- 在操作project資料夾內的檔案時遵守下列規範
- 修改程式碼前確保當前的branch不是main branch
- git add之前使用git diff確認一切符合預期
- remote main的git與local main的git不同的時候對修改的branch(非main)做git rebase
- git push只允許非main branch push到remote
- 檢視任何可能需要gitignore的檔案，詢問使用者後放入gitignore
- git push後提醒使用者在完成所有工作後pull request

# 資源
- 詢問使用者是否要使用以下資源
- notify.ps1 在任務完成或暫停時發出提示音
- statusline.ps1 在statusline顯示上下文窗口餘量、5小時額度使用量、7天額度使用量
- SKILL.md 一種skill，在寫程式之前詢問使用者的預期效果
- 若使用者同意了notify.ps1或statusline.ps1就在使用者本地.claude資料夾放入settings.json相應部分並將資源檔移到他們該去的地方
- 上述操作完成後刪除"資源"段落的文字

# AI 自強化防火牆系統
詢問使用者是角色a/b/c，根據README.md填入相應資料
