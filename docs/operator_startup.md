# 操作者启动入口

当前唯一日常步骤见 [真机操作指南](pick_a_steps.md)：

`start` → 保持 `console` 开启 → 移走桌子、空手 `prepare` → 放回桌子 → `spell ACE` →
空手、移走桌子后 `shutdown`。

prepare/recover/shutdown 均在输入 `CLEARED` 后实际执行动作，不拍桌面、不调用视觉 API。
旧版“prepare 只建模、不运动”的说明已废弃。仅路径检查通过不等于动作完成。

启动配置与相机标定是本地运行依赖，不能随失败报告一起删除。
`success_verified=false` 不等于运动失败：当前 spell 由操作者确认实际成功，程序记录动作是否完成。
