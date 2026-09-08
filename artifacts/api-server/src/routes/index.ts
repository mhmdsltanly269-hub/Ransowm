import { Router, type IRouter } from "express";
import healthRouter from "./health";
import ridsRouter from "./rids";

const router: IRouter = Router();

router.use(healthRouter);
router.use(ridsRouter);

export default router;
